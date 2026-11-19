import sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import torch
from dequant import dequant_awq, dequant_gptq, pack_awq
REV = [0, 4, 1, 5, 2, 6, 3, 7]

def unpack_awq(qw, qz, bits):
    sh = torch.arange(0, 32, bits)
    iw = torch.bitwise_right_shift(qw[:, :, None], sh[None, None, :]).to(torch.int8).view(qw.shape[0], -1)
    iz = torch.bitwise_right_shift(qz[:, :, None], sh[None, None, :]).to(torch.int8).view(qz.shape[0], -1)
    return (iw, iz)

def reverse(iw, iz, bits):
    r = torch.arange(iw.shape[-1], dtype=torch.int32).view(-1, 32 // bits)[:, REV].view(-1)
    return (iw[:, r], iz[:, r])

def dequantize_gemm(qw, qz, s, bits, g):
    iw, iz = reverse(*unpack_awq(qw, qz, bits), bits)
    iw = torch.bitwise_and(iw, 2 ** bits - 1)
    iz = torch.bitwise_and(iz, 2 ** bits - 1)
    return (iw - iz.repeat_interleave(g, dim=0)) * s.repeat_interleave(g, dim=0)

def pack_exllama(iw, iz, bits):
    sh = torch.arange(0, 32, bits)
    iw = iw.view(iw.shape[0] // (32 // bits), 32 // bits, -1)
    qw = torch.bitwise_left_shift(iw, sh[None, :, None]).sum(dim=1).to(torch.int32)
    iz = iz.view(-1, iz.shape[1] // (32 // bits), 32 // bits)
    qz = torch.bitwise_left_shift(iz, sh[None, None, :]).sum(dim=-1).to(torch.int32)
    return (qw, qz)

def unpack_reorder_pack(qw, qz, bits):
    iw, iz = reverse(*unpack_awq(qw, qz, bits), bits)
    iw = torch.bitwise_and(iw, 2 ** bits - 1)
    iz = torch.bitwise_and(iz, 2 ** bits - 1) - 1
    return pack_exllama(iw.int(), iz.int(), bits)
torch.manual_seed(1)
ok = True
for _ in range(5):
    inn, out, g = (512, 256, 128)
    q = torch.randint(0, 16, (inn, out))
    z = torch.randint(1, 16, (inn // g, out))
    s = (torch.rand(inn // g, out) * 0.02 + 0.001).half()
    qw, qz = pack_awq(q, z)
    ref = dequantize_gemm(qw, qz, s, 4, g)
    ok &= torch.equal(dequant_awq(qw, qz, s, 4, g), ref.t())
    gq, gz = unpack_reorder_pack(qw, qz, 4)
    ok &= torch.equal(dequant_gptq(gq, gz, s, None, 4, g), ref.t())
print('reference cross-check', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
