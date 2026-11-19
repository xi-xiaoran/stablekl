from __future__ import annotations
import torch
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE = [0, 4, 1, 5, 2, 6, 3, 7]

def _unpack_cols(packed, bits):
    n = 32 // bits
    sh = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    return (packed.unsqueeze(-1) >> sh & (1 << bits) - 1).reshape(packed.shape[0], packed.shape[1] * n)

def _unpack_rows(packed, bits):
    n = 32 // bits
    sh = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    return (packed.unsqueeze(1) >> sh.view(1, n, 1) & (1 << bits) - 1).reshape(packed.shape[0] * n, packed.shape[1])

def dequant_awq(qweight, qzeros, scales, bits=4, group=128):
    assert bits == 4, 'AWQ GEMM packing is defined here for 4 bits only'
    rev = torch.arange(qweight.shape[1] * 8, device=qweight.device).view(-1, 8)[:, AWQ_REVERSE].reshape(-1)
    q = _unpack_cols(qweight, bits)[:, rev]
    z = _unpack_cols(qzeros, bits)[:, rev]
    s = scales.float()
    g = torch.arange(q.shape[0], device=q.device) // group
    return ((q - z[g]).float() * s[g]).half().t().contiguous()

def dequant_gptq(qweight, qzeros, scales, g_idx=None, bits=4, group=128):
    q = _unpack_rows(qweight, bits)
    z = _unpack_cols(qzeros, bits) + 1 & (1 << bits) - 1
    gi = g_idx.long() if g_idx is not None else torch.arange(q.shape[0], device=q.device) // group
    return ((q - z[gi]).float() * scales.float()[gi]).half().t().contiguous()

def rtn(W, bits, group=None):
    out, inn = W.shape
    g = inn if group is None else group
    assert inn % g == 0
    Wf = W.float().reshape(out, inn // g, g)
    lo = Wf.amin(-1, keepdim=True)
    hi = Wf.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    s = ((hi - lo) / qmax).clamp(min=1e-10)
    z = torch.round(-lo / s).clamp(0, qmax)
    q = torch.clamp(torch.round(Wf / s) + z, 0, qmax)
    return ((q - z) * s).reshape(out, inn).half()

def pack_awq(q_in_out, z_g_out):

    def p(x):
        r, c = x.shape
        x = x.view(r, c // 8, 8)[:, :, AWQ_ORDER].int()
        out = torch.zeros(r, c // 8, dtype=torch.int32)
        for i in range(8):
            out |= x[:, :, i] << 4 * i
        return out
    return (p(q_in_out), p(z_g_out))

def pack_gptq(q_in_out, z_g_out, bits):
    n = 32 // bits
    r, c = q_in_out.shape
    qw = torch.zeros(r // n, c, dtype=torch.int32)
    x = q_in_out.int().view(r // n, n, c)
    for j in range(n):
        qw |= x[:, j, :] << bits * j
    zg = z_g_out.int() - 1
    gr, gc = zg.shape
    qz = torch.zeros(gr, gc // n, dtype=torch.int32)
    y = zg.view(gr, gc // n, n)
    for j in range(n):
        qz |= (y[:, :, j] & (1 << bits) - 1) << bits * j
    return (qw, qz)
