"""KL(softmax(teacher) || softmax(student)) computed from raw logits without forming probabilities; use it through stablekl.logits_kl."""
import math
import torch

def _round_once(v64, dt):
    if dt == torch.float64:
        return v64
    if dt == torch.float32:
        return v64.to(torch.float32)
    if dt in (torch.float16, torch.bfloat16):
        f = v64.to(torch.float32)
        back = f.to(torch.float64)
        inexact = back != v64
        bits = f.view(torch.int32)
        towards = torch.where(back.abs() > v64.abs(), bits - 1, bits)
        odd = torch.where(inexact, towards | 1, bits)
        f = torch.where(torch.isfinite(v64) & inexact, odd.view(torch.float32), f)
        return f.to(dt)
    raise TypeError(dt)
_PHI = [(k + 1) / math.factorial(k + 2) for k in range(16)]

def _phi(w):
    small = w.abs() < 0.1
    ws = torch.where(small, w, torch.zeros_like(w))
    acc = torch.full_like(w, _PHI[-1])
    for c in reversed(_PHI[:-1]):
        acc = acc * ws + c
    series = ws * ws * acc
    return torch.where(small, series, w * torch.exp(w) - torch.expm1(w))

class LogitsKL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, c):
        A, C = (a.detach().double(), c.detach().double())
        u = A - C
        q = torch.softmax(C, -1)
        delta = torch.log1p((q * torch.expm1(u)).sum(-1, keepdim=True))
        w = u - delta
        kl = (q * _phi(w)).sum(-1).mean()
        ctx.save_for_backward(q, w)
        ctx.dt = c.dtype
        ctx.B = A.shape[0]
        return _round_once(kl, a.dtype)

    @staticmethod
    def backward(ctx, G):
        q, w = ctx.saved_tensors
        g = -(G.double() / ctx.B) * q * torch.expm1(w)
        return (None, _round_once(g, ctx.dt))

def logits_kl(teacher_logits, student_logits):
    return LogitsKL.apply(teacher_logits, student_logits)
