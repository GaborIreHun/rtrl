from dataclasses import dataclass, InitVar
import gym

from rtrl.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper


@dataclass
class GymEnv(gym.Wrapper):
  seed_val: InitVar[int]  # the name seed is already taken by the gym.Env.seed function
  id: str = "Pendulum-v0"
  real_time: bool = False

  def __post_init__(self, seed_val):
    env = gym.make(self.id)
    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    # env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    # env = DictActionWrapper(env)
    if self.real_time:
      env = RealTimeWrapper(env)
    super().__init__(env)
    self.seed(seed_val)


