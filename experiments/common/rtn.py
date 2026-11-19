import torch

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
    return ((q - z) * s).reshape(out, inn)
START = {'S0': None, 'S1': (8, None), 'S2': (4, 128)}

def quantise_decoder_linears(model, bits, group):
    n = 0
    for name, m in model.named_modules():
        if name.startswith('model.layers.') and isinstance(m, torch.nn.Linear):
            g = group if group and m.weight.shape[1] % group == 0 else None
            m.weight.data = rtn(m.weight.data, bits, g).to(m.weight.dtype)
            n += 1
    if n == 0 or n % 7:
        raise SystemExit(f'expected 7 linear modules per decoder layer, found {n}')
    return n
