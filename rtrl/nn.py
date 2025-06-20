from copy import deepcopy
from dataclasses import InitVar, dataclass

import numpy as np
import torch
from torch.distributions import Distribution, Normal
from torch.nn import Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_, calculate_gain
from torch.nn.parameter import Parameter

from rtrl import partial


def no_grad(model):
  for p in model.parameters():
    p.requires_grad = False
  return model


@torch.no_grad()
def exponential_moving_average(averages, values, factor):
  """
  In-place exponential moving average: a ← (1-factor)*a + factor*v
  """
  for a, v in zip(averages, values):
    # this updates the tensor `a` itself
    a.copy_(a * (1 - factor) + v * factor)


def copy_shared(model_a):
  """Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`."""
  model_b = deepcopy(model_a)
  sda = model_a.state_dict(keep_vars=True)
  sdb = model_b.state_dict(keep_vars=True)
  for key in sda:
    a, b = sda[key], sdb[key]
    b.data = a.data  # strangely this will not make a.data and b.data the same object but their underlying data_ptr will be the same
    assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
  return model_b

class PopArt(torch.nn.Module):
    """PopArt normalization with safe, split update/rescale."""

    def __init__(self, output_layers, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8):
        super().__init__()
        self.output_layers = output_layers
        shape  = output_layers[0].bias.shape
        device = output_layers[0].bias.device

        self.beta        = beta
        self.zero_debias = zero_debias
        self.start_pop   = start_pop

        # running stats
        self.mean        = torch.nn.Parameter(torch.zeros(shape,     device=device), requires_grad=False)
        self.mean_square = torch.nn.Parameter(torch.ones(shape,     device=device), requires_grad=False)
        self.std         = torch.nn.Parameter(torch.ones(shape,     device=device), requires_grad=False)
        self.updates     = 0

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """Update running mean & std (no in-place on actor/critic weights)."""
        beta = max(1 / (self.updates + 1), self.beta) if self.zero_debias else self.beta

        new_mean        = (1 - beta) * self.mean        + beta * targets.mean(dim=0)
        new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(dim=0)
        new_std         = (new_mean_square - new_mean * new_mean).sqrt().clamp(min=1e-4)

        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates = self.updates + 1

    @torch.no_grad()
    def rescale_layers(self):
        """Rescale output layers to keep their outputs consistent."""
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                scale     = (self.std / self.std.detach())
                bias_term = (self.mean - self.mean.detach())
                layer.weight.copy_((layer.weight * scale.unsqueeze(-1)).detach())
                layer.bias  .copy_(((layer.bias + bias_term) * scale).detach())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Out‐of‐place normalization."""
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalized preds back to original scale."""
        return x * self.std + self.mean
    """PopArt normalization with safe, split update/rescale."""

    def __init__(self, output_layers, beta: float = 0.0003,
                 zero_debias: bool = True, start_pop: int = 8):
        super().__init__()
        self.output_layers = output_layers
        shape  = output_layers[0].bias.shape
        device = output_layers[0].bias.device

        self.beta         = beta
        self.zero_debias  = zero_debias
        self.start_pop    = start_pop
        self.mean         = torch.nn.Parameter(torch.zeros(shape,
                                         device=device), requires_grad=False)
        self.mean_square = torch.nn.Parameter(torch.ones(shape,
                                         device=device), requires_grad=False)
        self.std          = torch.nn.Parameter(torch.ones(shape,
                                         device=device), requires_grad=False)
        self.updates      = 0

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """
        Update running mean & std only (no in-place on autograd tensors).
        """
        beta = max(1 / (self.updates + 1), self.beta) \
               if self.zero_debias else self.beta

        # new first & second moments
        new_mean        = (1 - beta) * self.mean \
                          + beta * targets.mean(dim=0)
        new_mean_square = (1 - beta) * self.mean_square \
                          + beta * (targets * targets).mean(dim=0)
        new_std         = (new_mean_square - new_mean * new_mean) \
                          .sqrt().clamp(min=1e-4)

        # commit stats (no grad)
        self.mean       .copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std        .copy_(new_std)
        self.updates = self.updates + 1

    @torch.no_grad()
    def rescale_layers(self):
        """
        After stats have been updated, adjust each output layer so its
        outputs remain consistent. Call only when no graph is alive.
        """
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                scale     = (self.std / self.std.detach())
                bias_term = (self.mean - self.mean.detach())
                layer.weight.copy_((layer.weight * scale.unsqueeze(-1)).detach())
                layer.bias  .copy_(((layer.bias + bias_term) * scale).detach())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Out-of-place normalization using running stats."""
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalized preds back to original scale."""
        return x * self.std + self.mean


# noinspection PyAbstractClass
class TanhNormal(Distribution):
  # If passed in an out-of-range value (e.g. something outside −1,1−1,1 to the TanhNormal),
  # PyTorch wouldn’t raise a ValueError
  arg_constraints = {}
  """Distribution of X ~ tanh(Z) where Z ~ N(mean, std)
  Adapted from https://github.com/vitchyr/rlkit
  """
  def __init__(self, normal_mean, normal_std, epsilon=1e-6):
    self.normal_mean = normal_mean
    self.normal_std = normal_std
    self.normal = Normal(normal_mean, normal_std)
    self.epsilon = epsilon
    super().__init__(self.normal.batch_shape, self.normal.event_shape)

  def log_prob(self, x):
    assert hasattr(x, "pre_tanh_value")
    assert x.dim() == 2 and x.pre_tanh_value.dim() == 2
    return self.normal.log_prob(x.pre_tanh_value) - torch.log(
      1 - x * x + self.epsilon
    )

  def sample(self, sample_shape=torch.Size()):
    z = self.normal.sample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out

  def rsample(self, sample_shape=torch.Size()):
    z = self.normal.rsample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out


# noinspection PyAbstractClass
class Independent(torch.distributions.Independent):
  def sample_deterministic(self):
    return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(torch.nn.Module):
  def __init__(self, n, m):
    super().__init__()

    self.lin_mean = torch.nn.Linear(n, m)
    # self.lin_mean.weight.data
    # self.lin_mean.bias.data

    self.lin_std = torch.nn.Linear(n, m)
    self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
    self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

  def forward(self, x):
    mean = self.lin_mean(x)
    log_std = self.lin_std(x)
    log_std = torch.clamp(log_std, -20, 2)
    std = torch.exp(log_std)
    # a = TanhTransformedDist(Independent(Normal(m, std), 1))
    a = Independent(TanhNormal(mean, std), 1)
    return a


class RlkitLinear(torch.nn.Linear):
  def __init__(self, *args):
    super().__init__(*args)
    # TODO: investigate the following
    # this mistake seems to be in rlkit too
    # https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
    fan_in = self.weight.shape[0]  # this is actually fanout!!!
    bound = 1. / np.sqrt(fan_in)
    self.weight.data.uniform_(-bound, bound)
    self.bias.data.fill_(0.1)


class SacLinear(torch.nn.Linear):
  def __init__(self, in_features, out_features):
    super().__init__(in_features, out_features)
    with torch.no_grad():
      self.weight.uniform_(-0.06, 0.06)  # 0.06 == 1 / sqrt(256)
      self.bias.fill_(0.1)


class BasicReLU(torch.nn.Linear):
  def forward(self, x):
    x = super().forward(x)
    return torch.relu(x)


class AffineReLU(BasicReLU):
  def __init__(self, in_features, out_features, init_weight_bound: float = 1., init_bias: float = 0.):
    super().__init__(in_features, out_features)
    bound = init_weight_bound / np.sqrt(in_features)
    self.weight.data.uniform_(-bound, bound)
    self.bias.data.fill_(init_bias)


class NormalizedReLU(torch.nn.Sequential):
  def __init__(self, in_features, out_features, prenorm_bias=True):
    super().__init__(
      torch.nn.Linear(in_features, out_features, bias=prenorm_bias),
      torch.nn.LayerNorm(out_features),
      torch.nn.ReLU())


class KaimingReLU(torch.nn.Linear):
  def __init__(self, in_features, out_features):
    super().__init__(in_features, out_features)
    with torch.no_grad():
      kaiming_uniform_(self.weight)
      self.bias.fill_(0.)

  def forward(self, x):
    x = super().forward(x)
    return torch.relu(x)


Linear10 = partial(AffineReLU, init_bias=1.)
Linear04 = partial(AffineReLU, init_bias=0.4)
LinearConstBias = partial(AffineReLU, init_bias=0.1)
LinearZeroBias = partial(AffineReLU, init_bias=0.)
AffineSimon = partial(AffineReLU, init_weight_bound=0.01, init_bias=1.)


def dqn_conv(n):
  return torch.nn.Sequential(
      torch.nn.Conv2d(n, 32, kernel_size=8, stride=4),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
      torch.nn.ReLU()
    )


def big_conv(n):
  # if input shape = 64 x 256 then output shape = 2 x 26
  return torch.nn.Sequential(
    torch.nn.Conv2d(n, 64, 8, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 64, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 128, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(128, 128, 4, stride=1), torch.nn.LeakyReLU(),
  )


def hd_conv(n):
  return torch.nn.Sequential(
    torch.nn.Conv2d(n, 32, 8, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(32, 64, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 64, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 128, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(128, 128, 4, stride=2), torch.nn.LeakyReLU(),
  )