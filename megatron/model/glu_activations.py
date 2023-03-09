import torch
from torch import nn
from torch.nn import functional as F

from megatron import logging
from megatron.model.utils import log_debug_usage
from megatron import mpu

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


class _T5GLUBase(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            activation_fn=torch.sigmoid,
            bias=False,
            gather_output=True,
            init_method=torch.nn.init.xavier_normal_,
    ):
        super().__init__()
        self.linear = mpu.ColumnParallelLinear(
            in_features,
            out_features,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )
        self.nonlinear = mpu.ColumnParallelLinear(
            in_features,
            out_features,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )
        self.activation_fn = activation_fn

    def forward(self, x):
        output = self.linear(x)[0] * self.activation_fn(self.nonlinear(x)[0])
        return output, None


class T5LiGLU(_T5GLUBase):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            gather_output=True,
            init_method=torch.nn.init.xavier_normal_,
    ):
        super().__init__(
            in_features,
            out_features,
            activation_fn=nn.Identity(),
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )


class T5GEGLU(_T5GLUBase):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            gather_output=True,
            init_method=torch.nn.init.xavier_normal_,
    ):
        super().__init__(
            in_features,
            out_features,
            activation_fn=F.gelu,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )


class T5ReGLU(_T5GLUBase):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            gather_output=True,
            init_method=torch.nn.init.xavier_normal_,
    ):
        super().__init__(
            in_features,
            out_features,
            activation_fn=F.relu,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )


class T5SwiGLU(_T5GLUBase):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            gather_output=True,
            init_method=torch.nn.init.xavier_normal_,
    ):
        super().__init__(
            in_features,
            out_features,
            activation_fn=F.silu,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
        )


def replaces_linear(wrapped_glu_act):
    """Return whether the GLU activation wrapped by `log_debug_usage`
    contains a type.
    """
    return isinstance(wrapped_glu_act.__closure__[0].cell_contents, type)


liglu = log_debug_usage(logger, "Using GLU activation: LiGLU.")(torch.jit.script(LiGLU()))
geglu = log_debug_usage(logger, "Using GLU activation: GELU.")(torch.jit.script(GEGLU()))
reglu = log_debug_usage(logger, "Using GLU activation: ReGLU.")(torch.jit.script(ReGLU()))
swiglu = log_debug_usage(logger, "Using GLU activation: SwiGLU.")(torch.jit.script(SwiGLU()))
t5_liglu = log_debug_usage(logger, "Using GLU activation: T5LiGLU.")(T5LiGLU)
t5_geglu = log_debug_usage(logger, "Using GLU activation: T5GELU.")(T5GEGLU)
t5_reglu = log_debug_usage(logger, "Using GLU activation: T5ReGLU.")(T5ReGLU)
t5_swiglu = log_debug_usage(logger, "Using GLU activation: T5SwiGLU.")(T5SwiGLU)


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
    "t5_geglu": t5_geglu,
    "t5_liglu": t5_liglu,
    "t5_reglu": t5_reglu,
    "t5_swiglu": t5_swiglu,
}
