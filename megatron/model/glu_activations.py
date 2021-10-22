import torch
from torch import nn
from torch.nn import functional as F

from megatron import logging

logger = logging.get_logger(__name__)

class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.__logged_forward__ = False

    def forward(self, x):
        if self.__logged_forward__ is False:
            logger.debug("Using GLU activations.")
            self.__logged_forward__ = True
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(nn.Identity())


class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.silu)


liglu = torch.jit.script(LiGLU())
geglu = torch.jit.script(GEGLU())
reglu = torch.jit.script(ReGLU())
swiglu = torch.jit.script(SwiGLU())


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
}
