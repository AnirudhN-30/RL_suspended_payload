"""FLARE actor with the paper's final tanh projection."""

import torch
from rsl_rl.models import MLPModel
from torch import nn


class FlareTanhActor(MLPModel):
  """Bound the Gaussian mean while retaining RSL-RL's Gaussian exploration."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mlp.add_module("output_tanh", nn.Tanh())
    self.register_buffer("output_tanh_version", torch.tensor(1))


class FlarePayloadActorV2(FlareTanhActor):
  """Payload actor versioned for the quad/payload target-error observation."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.register_buffer("payload_observation_v2", torch.tensor(1))
