# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# Original implementation adjusted from https://github.com/sunyt32/torchscale

def fixed_pos_embedding(x, base):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.cos(sinusoid_inp), torch.sin(sinusoid_inp)


class XPos(torch.nn.Module):
    """
    xPos positional embeddings from https://arxiv.org/abs/2212.10554.
    """

    def __init__(self, head_dim, freq_base=10000, scale_base=512, gamma=0.4, precision=torch.half):
        super().__init__()
        self.scale_base = scale_base
        self.register_buffer(
            "scale",
            (
                (torch.arange(0, head_dim, 2) + gamma * head_dim)
                / ((1.0 + gamma) * head_dim)
            ),
        )
        self.max_seq_len_cached = None
        self.precision = precision
        self.freq_base = freq_base

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if (
                self.max_seq_len_cached is None
                or (seq_len > self.max_seq_len_cached)
        ):
            self.max_seq_len_cached = seq_len
            scale = (
                self.scale
                ** (
                    torch.arange(0, seq_len, 1) - seq_len // 2
                ).to(self.scale).div(self.scale_base)[:, None]
            )
            cos, sin = fixed_pos_embedding(scale, self.freq_base)
            self.cos_cached = cos
            self.sin_cached = sin
            self.scale_cached = scale
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
            self.scale_cached[:seq_len],
        )


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m.unsqueeze(1)


def _apply_xpos_emb(x, cos, sin, scale):
    # x is assumed to be (seq_len, batch_size, dim) here.
    cos = duplicate_interleave(cos * scale)
    sin = duplicate_interleave(sin * scale)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


@torch.jit.script
def apply_xpos_emb(q, k, cos, sin, scale, offset: int = 0):
    # q/k are assumed to be (seq_len, batch_size, dim) here.
    cos = cos[offset:q.shape[0] + offset]
    sin = sin[offset:q.shape[0] + offset]
    scale = scale[offset:q.shape[0] + offset]
    return (
        _apply_xpos_emb(q, cos, sin, scale),
        _apply_xpos_emb(q, cos, sin, 1.0 / scale),
    )


def apply_xpos_emb_torch(q, k, cos, sin, scale, offset: int = 0):
    # q/k are assumed to be (seq_len, batch_size, dim) here.
    cos = cos[offset:q.shape[0] + offset]
    sin = sin[offset:q.shape[0] + offset]
    scale = scale[offset:q.shape[0] + offset]
    return (
        _apply_xpos_emb(q, cos, sin, scale),
        _apply_xpos_emb(q, cos, sin, 1.0 / scale),
    )
