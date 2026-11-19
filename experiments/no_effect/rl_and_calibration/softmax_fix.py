import torch, torch.nn.functional as F

def fixed_log_softmax(z, dim=-1):
    idx = z.argmax(dim=dim, keepdim=True)
    d = z - z.gather(dim, idx)
    zero = torch.zeros_like(d.gather(dim, idx))
    zt = d.scatter(dim, idx, zero)
    others = d.scatter(dim, idx, zero - float('inf'))
    m = -torch.logsumexp(others, dim=dim, keepdim=True)
    return zt - F.softplus(-m, threshold=float('inf'))

def fixed_softmax(z, dim=-1):
    return fixed_log_softmax(z, dim).exp()
