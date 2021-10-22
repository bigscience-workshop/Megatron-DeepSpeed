import torch
from torch import nn
from torch.nn import functional as F

from megatron import logging

logger = logging.get_logger(__name__)

class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn, logger):
        super().__init__()
        self.activation_fn = activation_fn
        self.logger = logger
        self._logged_forward = False

    def forward(self, x):
        if self._logged_forward is False:
            self.logger.debug("Using GLU activations.")
            self._logged_forward = True
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    def __init__(self, logger):
        super().__init__(nn.Identity(), logger)


class GEGLU(_GLUBaseModule):
    def __init__(self, logger):
        super().__init__(F.gelu, logger)


class ReGLU(_GLUBaseModule):
    def __init__(self, logger):
        super().__init__(F.relu, logger)


class SwiGLU(_GLUBaseModule):
    def __init__(self, logger):
        super().__init__(F.silu, logger)


liglu = torch.jit.script(LiGLU(logger))
geglu = torch.jit.script(GEGLU(logger))
reglu = torch.jit.script(ReGLU(logger))
swiglu = torch.jit.script(SwiGLU(logger))


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
}
