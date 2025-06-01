from collections.abc import Sequence, Mapping
from collections import deque
import gym
import numpy as np


class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, env.action_space.low]),
            high=np.concatenate([env.observation_space.high, env.action_space.high]),
            dtype=np.float32
        )
        assert isinstance(env.action_space, gym.spaces.Box)
        self.initial_action = np.zeros_like(env.action_space.low, dtype=np.float32)
        self.previous_action = self.initial_action  # Initialize self.previous_action
        print("Previous action shape:", self.previous_action.shape)

    def reset(self):
        self.previous_action = self.initial_action
        result = super().reset()
        if isinstance(result, tuple):  # Handle new Gym API
            observation, info = result
        else:
            observation = result
            info = {}
        print("Observation shape:", np.asarray(observation, dtype=np.float32).shape)
        return np.concatenate([np.asarray(observation, dtype=np.float32), self.previous_action])

    def step(self, action):
        # 1) step the underlying env
        result = self.env.step(action)
        # 2) unpack new/old Gym API
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            observation, reward, done, info = result
        else:
            raise RuntimeError(f"Unexpected return from env.step(): {len(result)}")

        # 3) record this action as "previous" for next call
        self.previous_action = np.asarray(action, dtype=np.float32)

        # 4) concatenate obs + previous_action just like in reset()
        obs_combined = np.concatenate(
            [np.asarray(observation, dtype=np.float32), self.previous_action]
        )
        return obs_combined, reward, done, info

class PreviousActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, env.action_space.low]),
            high=np.concatenate([env.observation_space.high, env.action_space.high]),
            dtype=np.float32
        )
        assert isinstance(env.action_space, gym.spaces.Box)
        self.initial_action = np.zeros_like(env.action_space.low, dtype=np.float32)

    def reset(self):
        self.previous_action = self.initial_action
        result = super().reset()
        if isinstance(result, tuple):  # Handle new Gym API
            observation, info = result
        else:
            observation = result
            info = {}
        return np.concatenate([np.asarray(observation, dtype=np.float32), self.previous_action])

    def step(self, action):
        result = super().step(action)
        if len(result) == 5:
            observation, reward, done, truncated, info = result
            done = done or truncated
        elif len(result) == 4:
            observation, reward, done, info = result
        else:
            raise RuntimeError(f"Unexpected number of elements returned from env.step: {len(result)}")
        self.previous_action = np.asarray(action, dtype=np.float32)
        return np.concatenate([np.asarray(observation, dtype=np.float32), self.previous_action]), reward, done, info    


class StatsWrapper(gym.Wrapper):
  """Compute running statistics (return, number of episodes, etc.) over a certain time window."""

  def __init__(self, env, window=100):
    super().__init__(env)
    self.reward_hist = deque([0], maxlen=window + 1)
    self.done_hist = deque([1], maxlen=window + 1)
    self.total_steps = 0

  def reset(self, **kwargs):
    return super().reset(**kwargs)

  def step(self, action):
    # call the next wrapper/env
    result = super().step(action)
    # handle new (5-tuple) vs. old (4-tuple) Gym APIs
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
      done = terminated or truncated
    elif len(result) == 4:
      obs, reward, done, info = result
    else:
      raise RuntimeError(f"Unexpected number of elements returned from env.step: {len(result)}")

    # record statistics
    self.reward_hist.append(reward)
    self.done_hist.append(done)
    self.total_steps += 1

    # always return obs, reward, done, info
    return obs, reward, done, info

  def stats(self):
    returns = [0]
    steps = [0]
    for reward, done in zip(self.reward_hist, self.done_hist):
      returns[-1] += reward
      steps[-1] += 1
      if done:
        returns.append(0)
        steps.append(0)
    returns = returns[1:-1]  # first and last episodes are incomplete
    steps = steps[1:-1]

    return dict(
      episodes=len(returns),
      episode_length=np.mean(steps) if len(steps) else np.nan,
      returns=np.mean(returns) if len(returns) else np.nan,
      average_reward=np.mean(tuple(self.reward_hist)[1:]),
    )


class DictObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, key='vector'):
    super().__init__(env)
    self.key = key
    self.observation_space = gym.spaces.Dict({self.key: env.observation_space})

  def observation(self, observation):
    return {self.key: observation}


class TupleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space,))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
            return (self.observation(observation), info)
        else:
            return self.observation(result)
        
    def step(self, action):
        # call through to the wrapped env
        result = self.env.step(action)
        # support both new (5-tuple) and old (4-tuple) Gym APIs
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            observation, reward, done, info = result
        else:
            raise RuntimeError(f"Unexpected return from env.step(): {len(result)}")

        # wrap the obs in a 1-tuple just like reset()
        wrapped_obs = (observation,)
        return wrapped_obs, reward, done, info

    def observation(self, observation):
        return (observation,)


class DictActionWrapper(gym.Wrapper):
  def __init__(self, env, key='value'):
    super().__init__(env)
    self.key = key
    self.action_space = gym.spaces.Dict({self.key: env.action_space})

  def step(self, action: dict):
    return self.env.step(action['value'])


class AffineObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, shift, scale):
    super().__init__(env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    self.shift = shift
    self.scale = scale
    self.observation_space = gym.spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high), dtype=env.observation_space.dtype)

  def observation(self, obs):
    return (obs + self.shift) * self.scale


class AffineRewardWrapper(gym.RewardWrapper):
  def __init__(self, env, shift, scale):
    super().__init__(env)
    self.shift = shift
    self.scale = scale

  def reward(self, reward):
    return (reward + self.shift) / self.scale


class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scale = env.action_space.high - env.action_space.low
        self.shift = env.action_space.low
        self.action_space = gym.spaces.Box(-np.ones_like(self.shift), np.ones_like(self.shift), dtype=env.action_space.dtype)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = action / 2 + 0.5  # 0 < a < 1
        action = action * self.scale + self.shift
        result = self.env.step(action)
        if len(result) == 5:
            return result
        elif len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        else:
            raise RuntimeError(f"Unexpected number of elements returned from env.step: {len(result)}")


class TimeLimitResetWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=None, key='reset'):
        super().__init__(env)
        self.reset_key = key
        from gym.wrappers import TimeLimit
        self.enforce = bool(max_steps)
        if max_steps is None:
            tl = get_wrapper_by_class(env, TimeLimit)
            max_steps = 1 << 31 if tl is None else tl._max_episode_steps
        self.max_steps = max_steps
        self.t = 0

    def reset(self, **kwargs):
        m = self.env.reset(**kwargs)
        self.t = 0
        return m

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            m, r, terminated, truncated, info = result
            d = terminated or truncated
        else:
            m, r, d, info = result
        reset = (self.t == self.max_steps - 1) or info.get(self.reset_key, False)
        if not self.enforce:
            if reset:
                assert d, f"something went wrong t={self.t}, max_steps={self.max_steps}, info={info}"
        else:
            d = d or reset
        info = {**info, self.reset_key: reset}
        self.t += 1
        return m, r, d, info


class Float64ToFloat32(gym.ObservationWrapper):
    """Converts np.float64 arrays in the observations to np.float32 arrays."""

    def observation(self, observation):
        observation = deepmap({np.ndarray: float64_to_float32}, observation)
        return observation

    def step(self, action):
        result = super().step(action)
        if len(result) == 5:
            s, r, terminated, truncated, info = result
            d = terminated or truncated
        else:
            s, r, d, info = result
        return s, r, d, info


# === Utilities ========================================================================================================

def get_wrapper_by_class(env, cls):
  if isinstance(env, cls):
    return env
  elif isinstance(env, gym.Wrapper):
    return get_wrapper_by_class(env.env, cls)


def deepmap(f, m):
  """Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
  Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
  for cls in f:
    if isinstance(m, cls):
      return f[cls](m)
  if isinstance(m, Sequence):
    return type(m)(deepmap(f, x) for x in m)
  elif isinstance(m, Mapping):
    return type(m)((k, deepmap(f, m[k])) for k in m)
  else:
    raise AttributeError()


def float64_to_float32(x):
    return np.asarray(x, np.float32) if x.dtype == np.float64 else x
