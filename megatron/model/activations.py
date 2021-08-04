from torch import nn
from torch.nn import functional as F


class _GLUVariant(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
    
    def forward(self, x, bias=None):
        x1, x2 = x.chunk(2, dim=-1)
        if bias is not None:
            b1, b2 = bias.chunk(2, dim=-1)
            x1 = x1 + b1
            x2 = x2 + b2
        return x1 * self.activation_fn(x2)


class Bilinear(_GLUVariant):
    def __init__(self):
        super().__init__(lambda x: x)


class GEGLU(_GLUVariant):
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUVariant):
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUVariant):
    def __init__(self):
        super().__init__(F.silu)
