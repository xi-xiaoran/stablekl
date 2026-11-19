import sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import torch
from dequant import *
torch.manual_seed(0)
ok = True
for trial in range(5):
    inn, out, g = (256, 384, 128)
    q = torch.randint(0, 16, (inn, out))
    z = torch.randint(0, 16, (inn // g, out))
    s = (torch.rand(inn // g, out) * 0.02 + 0.001).half()
    ref = ((q - z.repeat_interleave(g, 0)).float() * s.float().repeat_interleave(g, 0)).half().t()
    qw, qz = pack_awq(q, z)
    ok &= torch.equal(dequant_awq(qw, qz, s, 4, g), ref)
    for bits in (4, 8):
        qq = torch.randint(0, 1 << bits, (inn, out))
        zz = torch.randint(0, 1 << bits, (inn // g, out))
        ref = ((qq - zz.repeat_interleave(g, 0)).float() * s.float().repeat_interleave(g, 0)).half().t()
        qw, qz = pack_gptq(qq, zz, bits)
        ok &= torch.equal(dequant_gptq(qw, qz, s, None, bits, g), ref)
        gidx = torch.arange(inn) // g
        ok &= torch.equal(dequant_gptq(qw, qz, s, gidx, bits, g), ref)
W = torch.randn(64, 256)
for bits, gr in ((8, None), (4, 128), (3, 128)):
    Wq = rtn(W, bits, gr)
    levels = Wq.float().reshape(64, -1, gr or 256)
    ok &= all((len(torch.unique(levels[i, j])) <= 1 << bits for i in range(64) for j in range(levels.shape[1])))
    ok &= (Wq.float() - W).abs().max().item() < (W.max() - W.min()).item() / ((1 << bits) - 1)
print('dequant tests', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
