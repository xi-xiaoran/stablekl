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

_TINY = 2.3e-308      # below this a float64 probability is subnormal or zero
_BIG = 30.0           # above this q * expm1(x) is formed as exp(log q + x) - q


def _guarded(A, C, u, lo, q, w, bad):
    """log(p / q) for the guarded rows.  The difference is first centred at the student's largest logit, which
    is exact for a common offset (a = c + const gives w = 0), and the low part lost by the subtraction is added
    back; if 1 + sum q expm1(.) still cancels, it is centred once more by the offset just computed.  Where the
    difference of one category is large, q exp(.) is formed as exp(log q + .), which neither overflows nor loses
    a student probability that underflowed ('over' marks these entries, 'pw' holds p there).  Rows that still
    overflow ('far') use sum p (log p - log q) and the gradient q - p."""
    lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    lq = torch.log_softmax(C, -1)

    def qem1(x):
        return torch.where(x > _BIG, torch.exp(lq + x) - q, q * torch.expm1(x))

    uc = (u - torch.gather(u, -1, C.argmax(-1, keepdim=True))) + lo
    S = qem1(uc).sum(-1, keepdim=True)
    d = torch.log1p(S)
    again = (S < -0.5) & torch.isfinite(d)
    uc2 = torch.where(again, uc - d, uc)
    S2 = qem1(uc2).sum(-1, keepdim=True)
    wc = torch.where(again, uc2 - torch.log1p(S2), uc - d)
    ok = torch.isfinite(wc).all(-1)
    far = bad & ~ok
    w = torch.where((bad & ok).unsqueeze(-1), wc, w)
    over = (bad & ok).unsqueeze(-1) & (w > _BIG)
    pw = torch.exp(lq + w)
    p = torch.softmax(A, -1)
    lr = torch.log_softmax(A, -1) - lq
    row_far = torch.where(p > 0, p * lr, torch.zeros_like(p)).sum(-1)
    w = torch.where(far.unsqueeze(-1), torch.zeros_like(w), w)
    return w, over, pw, far, row_far, q - p


class LogitsKL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, c):
        A, C = (a.detach().double(), c.detach().double())
        u = A - C
        q = torch.softmax(C, -1)
        S = (q * torch.expm1(u)).sum(-1, keepdim=True)
        delta = torch.log1p(S)
        w = u - delta
        # Guarded rows: the float64 difference u lost bits (lo != 0), 1 + S cancels (the teacher's logits lie
        # below the student's by a common offset), or an intermediate overflowed.
        bb = u - A
        lo = (A - (u - bb)) + (-C - bb)
        bad = ((lo != 0) | ~torch.isfinite(w)).any(-1) | (S.squeeze(-1) < -0.5) | ((q < _TINY) & (u > 0)).any(-1)
        far = None
        if bool(bad.any()):
            w, over, pw, far, row_far, g_far = _guarded(A, C, u, lo, q, w, bad)
        terms = q * _phi(w)
        if far is not None:
            terms = torch.where(over, pw * (w - 1) + q, terms)
        rows = terms.sum(-1)
        if far is not None:
            rows = torch.where(far, row_far, rows)
        ctx.far, ctx.g_far, ctx.over, ctx.pw = (far, g_far, over, pw) if far is not None else (None,) * 4
        kl = rows.mean()
        ctx.save_for_backward(q, w)
        ctx.dt = c.dtype
        ctx.B = A.shape[0]
        return _round_once(kl, a.dtype)

    @staticmethod
    def backward(ctx, G):
        q, w = ctx.saved_tensors
        g = -(G.double() / ctx.B) * q * torch.expm1(w)
        if ctx.far is not None:
            g = torch.where(ctx.over, G.double() / ctx.B * (q - ctx.pw), g)
            g = torch.where(ctx.far.unsqueeze(-1), G.double() / ctx.B * ctx.g_far, g)
        return (None, _round_once(g, ctx.dt))

def logits_kl(teacher_logits, student_logits):
    return LogitsKL.apply(teacher_logits, student_logits)
