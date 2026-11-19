"""Single, correct rounding of float64 values to a narrower floating-point format.

``torch``'s direct cast from float64 to float16 or bfloat16 may round twice (through float32).  Rounding first to
float32 *to odd* and then to the narrow format gives the correctly rounded result, because float32 carries more
than two extra significand bits.  float64 -> float32 is a single correct rounding in torch."""
import torch

SUPPORTED = (torch.float16, torch.float32, torch.float64)


def round_to(v64, dtype):
    """Correctly rounded (round-to-nearest-even) conversion of a float64 tensor to ``dtype``."""
    if dtype == torch.float64:
        return v64
    if dtype == torch.float32:
        return v64.to(torch.float32)
    if dtype in (torch.float16, torch.bfloat16):
        f = v64.to(torch.float32)
        back = f.to(torch.float64)
        inexact = back != v64
        bits = f.view(torch.int32)
        towards_zero = torch.where(back.abs() > v64.abs(), bits - 1, bits)
        odd = torch.where(inexact, towards_zero | 1, bits)
        f = torch.where(torch.isfinite(v64) & inexact, odd.view(torch.float32), f)
        return f.to(dtype)
    raise TypeError(f"unsupported dtype {dtype}")


class Widen(torch.autograd.Function):
    """x -> float64 (exact); the incoming float64 gradient is rounded once to the dtype of x."""

    @staticmethod
    def forward(ctx, x):
        ctx.dt = x.dtype
        return x.to(torch.float64)

    @staticmethod
    def backward(ctx, g):
        return round_to(g.to(torch.float64), ctx.dt)


class Narrow(torch.autograd.Function):
    """float64 value -> dtype with one correct rounding; the gradient passes through exactly."""

    @staticmethod
    def forward(ctx, v64, dtype):
        return round_to(v64, dtype)

    @staticmethod
    def backward(ctx, g):
        return g.to(torch.float64), None
