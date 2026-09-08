"""FLARE actor with a bounded Gaussian mean, not a squashed Gaussian."""

import torch
from rsl_rl.models import MLPModel
from torch import nn


class FlareTanhActor(MLPModel):
  """Apply tanh before Gaussian sampling and during deterministic inference."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mlp.add_module("output_tanh", nn.Tanh())
    # Prevent strict loading of old linear-output actors with different semantics.
    self.register_buffer("output_tanh_version", torch.tensor(1))
