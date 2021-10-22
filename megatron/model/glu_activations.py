from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from megatron import logging

logger = logging.get_logger(__name__)

class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
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

def log_debug_usage(func, msg: str):
    func.__logged_message__ = False
    @wraps(func)
    def wrapped(*args, **kwargs):
        if func.__logged_message__ is False:
            logger.debug(msg)
            func.__logged_message__ = True
        return func(*args, **kwargs)
    return wrapped


liglu = log_debug_usage(torch.jit.script(LiGLU()), "Using GLU activation: LiGLU.")
geglu = log_debug_usage(torch.jit.script(GEGLU()), "Using GLU activation: GELU.")
reglu = log_debug_usage(torch.jit.script(ReGLU()), "Using GLU activation: ReGLU.")
swiglu = log_debug_usage(torch.jit.script(SwiGLU()), "Using GLU activation: SwiGLU.")


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
}
