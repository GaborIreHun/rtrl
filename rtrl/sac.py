from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache, reduce
from itertools import chain
import numpy as np
import torch
from torch.nn.functional import mse_loss

from rtrl.memory import Memory
from rtrl.nn import PopArt, no_grad, copy_shared, exponential_moving_average, hd_conv
from rtrl.util import cached_property, partial
import rtrl.sac_models

DEBUGGER = True

@dataclass(eq=0)
class Agent:
  observation_space: InitVar
  action_space: InitVar

  Model: type = rtrl.sac_models.Mlp
  OutputNorm: type = PopArt
  batchsize: int = 256  # training batch size
  memory_size: int = 1000000  # replay memory size
  lr: float = 0.0003  # learning rate
  discount: float = 0.99  # reward discount factor
  target_update: float = 0.005  # parameter for exponential moving average
  reward_scale: float = 5.
  entropy_scale: float = 1.
  start_training: int = 10000
  device: str = None
  training_interval: int = 1

  model_nograd = cached_property(lambda self: no_grad(deepcopy(self.model)))

  num_updates = 0
  training_steps = 0

  def __post_init__(self, observation_space, action_space):
    device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = self.Model(observation_space, action_space)
    self.model = model.to(device)
    self.model_target = no_grad(deepcopy(self.model))

    self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr)
    self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr)
    self.memory = Memory(self.memory_size, self.batchsize, device)

    self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
    self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

  def act(self, obs, r, done, info, train=False):
    # Ensure obs is a copy, not a view or reused array
    if isinstance(obs, np.ndarray):
        obs = np.copy(obs)
    stats = []
    action, _ = self.model.act(obs, r, done, info, train)

    if train:
      self.memory.append(np.float32(r), np.float32(done), info, obs, action)
      if len(self.memory) >= self.start_training and self.training_steps % self.training_interval == 0:
        stats = stats + self.train(),
      self.training_steps = self.training_steps + 1
    return action, stats

  def train(self):
    obs, actions, rewards, next_obs, terminals = self.memory.sample()
    rewards, terminals = rewards[:, None], terminals[:, None]

    # ---- Sample new actions ----
    new_dist = self.model.actor(obs)
    new_actions = new_dist.rsample()
    if DEBUGGER:
        print("new_actions version:", new_actions._version)

    # ---- Compute target Q via target networks ----
    next_dist = self.model_nograd.actor(next_obs)
    next_actions = next_dist.rsample()
    if DEBUGGER:
        print("next_actions version:", next_actions._version)

    next_values = [c(next_obs, next_actions) for c in self.model_target.critics]
    next_value = reduce(torch.min, next_values)
    if DEBUGGER:
        print("next_value version:", next_value._version)

    # Unnormalize, subtract entropy
    next_value = self.outputnorm_target.unnormalize(next_value)
    if DEBUGGER:
        print("next_value after unnormalize version:", next_value._version)

    next_value = next_value - self.entropy_scale * next_dist.log_prob(next_actions)[:, None]
    if DEBUGGER:
        print("next_value after entropy version:", next_value._version)

    # 1) build the raw Bellman target
    value_target = self.reward_scale * rewards \
                 + (1.0 - terminals) * self.discount * next_value
    if DEBUGGER:
        print("value_target version:", value_target._version)

    # detach before stats so no gradient flows into targets
    value_target_detached = value_target.detach()

    # update running mean/std (no_grad) then get a fresh normalized copy
    self.outputnorm.update_stats(value_target_detached)
    value_target_norm = self.outputnorm.normalize(value_target_detached)
    if DEBUGGER:
        print("value_target_norm version:", value_target_norm._version)

    # ---- Critic loss ----
    values = [c(obs, actions) for c in self.model.critics]
    assert values[0].shape == value_target.shape
    loss_critic = sum(mse_loss(v, value_target_norm) for v in values)

    # ---- Actor loss ----
    # evaluate Q under the new policy, then unnormalize
    new_vals = [c(obs, new_actions.detach()) for c in self.model.critics]
    new_val  = reduce(torch.min, new_vals)
    new_val  = self.outputnorm.unnormalize(new_val)

    raw_lp = new_dist.log_prob(new_actions)[:, None]
    loss_actor = self.entropy_scale * raw_lp - new_val
    assert loss_actor.shape == (self.batchsize, 1)
    loss_actor = self.outputnorm.normalize(loss_actor).mean()

    # ---- Optimize critic ----
    self.critic_optimizer.zero_grad()
    loss_critic.backward()
    self.critic_optimizer.step()

    # ---- Optimize actor ----
    self.actor_optimizer.zero_grad()
    if DEBUGGER:
        print("DEBUG new_val version:", new_val._version)
        print("DEBUG raw_lp version:", raw_lp._version)
        print("DEBUG loss_actor version:", loss_actor._version)
    loss_actor.backward()
    self.actor_optimizer.step()

    # ---- Soft‐update targets ----
    # ---- Soft‐update both network and normalizer targets ----
    exponential_moving_average(
        self.model_target.critics.parameters(),
        self.model.critics.parameters(),
        self.target_update
    )
    exponential_moving_average(
        self.outputnorm_target.parameters(),
        self.outputnorm.parameters(),
        self.target_update
    )

    return dict(
        loss_actor=loss_actor.item(),
        loss_critic=loss_critic.item(),
        outputnorm_mean=float(self.outputnorm.mean),
        outputnorm_std=float(self.outputnorm.std),
        memory_size=len(self.memory),
    )




AvenueAgent = partial(
  Agent,
  entropy_scale=0.05,
  lr=0.0002,
  memory_size=500000,
  batchsize=100,
  training_interval=4,
  start_training=10000,
  Model=partial(rtrl.sac_models.ConvModel)
)


# === tests ============================================================================================================
def test_agent():
  from rtrl import Training, run
  Sac_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=100,
    Agent=partial(Agent, memory_size=1000000, start_training=256, batchsize=4),
    Env=partial(id="Pendulum-v1", real_time=0),
  )
  run(Sac_Test)


def test_agent_avenue():
  from rtrl import Training, run
  from rtrl.envs import AvenueEnv
  Sac_Avenue_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=300,
    Agent=partial(AvenueAgent, device='cpu', training_interval=4, start_training=400),
    Env=partial(AvenueEnv, real_time=0),
    Test=partial(number=0),  # laptop can't handle more than that
  )
  run(Sac_Avenue_Test)


def test_agent_avenue_hd():
  from rtrl import Training, run
  from rtrl.envs import AvenueEnv
  Sac_Avenue_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=300,
    Agent=partial(AvenueAgent, device='cpu', training_interval=4, start_training=400, Model=partial(Conv=hd_conv)),
    Env=partial(AvenueEnv, real_time=0, width=368, height=368),
    Test=partial(number=0),  # laptop can't handle more than that
  )
  run(Sac_Avenue_Test)


if __name__ == "__main__":
  test_agent()
  # test_agent_avenue()
  # test_agent_avenue_hd()
