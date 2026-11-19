from __future__ import annotations
import argparse
import json
import multiprocessing as multiprocessing
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import numpy as np
from ..fp import get_format, round_fraction
from ..pipeline_io import load_unit, project_root, save_unit, sha256, text_array
from ..gen.constants import CONFIG_DISTRIBUTIONS
UNSUPPORTED_CPU_PATTERNS = (re.compile('not implemented for [\'\\"]?(?:Half|BFloat16|CPU)', re.I), re.compile('(?:unsupported|not available).*\\bcpu\\b', re.I))

def _torch():
    import torch
    return torch

def _dtype_np(dtype: str):
    return np.float16 if dtype == 'fp16' else np.float32

def _bits_to_tensor(bits, dtype, torch):
    bit_type = np.uint16 if dtype == 'fp16' else np.uint32
    values = np.ascontiguousarray(bits, dtype=bit_type)
    fp = values.view(_dtype_np(dtype))
    result = torch.from_numpy(fp.copy())
    assert np.array_equal(result.detach().cpu().numpy().view(bit_type), values), 'input bit-pattern round trip failed'
    return result

def _native_hex(value):
    tensor = value.detach().reshape(-1).cpu().contiguous()
    native = tensor.numpy()
    bit_dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[native.dtype.itemsize]
    return [format(int(b), f'0{native.dtype.itemsize * 2}x') for b in native.view(bit_dtype)]

def _round_output(value, dtype, method):
    import math
    torch = _torch()
    native_bits = _native_hex(value)
    native_dtype = str(value.dtype).replace('torch.', '')
    out = []
    fmt = get_format(dtype)
    for item, raw in zip(value.reshape(-1), native_bits):
        v = float(item.detach().cpu().item())
        if method == 'A':
            out.append(int(raw, 16))
        elif math.isnan(v):
            out.append(fmt.infinity_bits | 1)
        elif math.isinf(v):
            out.append(fmt.infinity_bits | (fmt.sign_mask if v < 0 else 0))
        else:
            out.append(round_fraction(Fraction(v), fmt))
    return (out, native_bits, native_dtype)

def _construct(config, tensors, torch):
    cls = getattr(torch.distributions, CONFIG_DISTRIBUTIONS[config])
    if config in ('binomial_probs', 'binomial_logits'):
        return cls(**tensors)
    return cls(**tensors)

def _capture_cached(dist, row_number):
    torch = _torch()
    output = {}
    for name, value in dist.__dict__.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            try:
                part = value.detach().cpu()
                if part.ndim and len(part) > row_number:
                    part = part[row_number]
                output[name] = {'dtype': str(part.dtype), 'shape': list(part.shape), 'bits': _native_hex(part)}
            except (TypeError, ValueError, RuntimeError) as exc:
                output[name] = {'error': f'{type(exc).__name__}: {exc}'}
    return json.dumps(output, sort_keys=True)

def _unsupported(exc):
    msg = str(exc)
    return any((pattern.search(msg) for pattern in UNSUPPORTED_CPU_PATTERNS))

def _call_batch(config, dtype, method, frozen, indices, device):
    torch = _torch()
    from torch.distributions import kl as kl_module
    low = {'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    compute = low if method == 'A' else torch.float32 if method == 'B32' else torch.float64
    spec = frozen['_parameter_shapes']
    tensors = {}
    for side in ('P', 'Q'):
        params = {}
        for name, desc in spec.items():
            raw = frozen[f'{side.lower()}__{name}'][indices]
            t = _bits_to_tensor(raw, dtype, torch)
            shape = tuple(desc['shape'])
            t = t.reshape((len(indices),) + shape)
            if method != 'A':
                t = t.to(dtype=compute)
            t = t.to(device=device).detach()
            if desc['differentiable']:
                t.requires_grad_(True)
            params[name] = t
        tensors[side] = params
    try:
        p = _construct(config, tensors['P'], torch)
        q = _construct(config, tensors['Q'], torch)
        registered = kl_module._KL_REGISTRY.get((type(p), type(q)))
        dispatched = kl_module._dispatch_kl(type(p), type(q))
        if registered is None or dispatched is not registered:
            raise AssertionError(f'KL dispatch mismatch: {type(p)} / {type(q)}')
        value = kl_module.kl_divergence(p, q)
        if value is None:
            raise ValueError('kl_divergence returned None')
        value = value.reshape(len(indices), -1).sum(dim=1)
    except Exception as exc:
        status = 'UNSUPPORTED_CPU' if _unsupported(exc) else 'CALL_ERROR'
        return [{'status': status, 'error': f'{type(exc).__name__}: {exc}'} for _ in indices]
    try:
        value.sum().backward()
        backward_error = None
        backward_status = None
    except Exception as exc:
        backward_error = f'{type(exc).__name__}: {exc}'
        backward_status = 'BACKWARD_UNSUPPORTED_CPU' if _unsupported(exc) else 'BACKWARD_ERROR'
    rounded, native, dtype_name = _round_output(value, dtype, method)
    out = []
    for j, _ in enumerate(indices):
        row = {'status': 'OK' if backward_error is None else backward_status, 'error': backward_error or '', 'kl_bits': rounded[j], 'kl_native_bits': native[j], 'native_dtype': dtype_name, 'cached_p': _capture_cached(p, j), 'cached_q': _capture_cached(q, j), 'grad': {}}
        for side in ('P', 'Q'):
            for name, desc in spec.items():
                if not desc['differentiable']:
                    continue
                leaf = tensors[side][name]
                size = int(np.prod(desc['shape']))
                grad = leaf.grad
                key = f'{side}__{name}'
                if backward_error is not None:
                    row['grad'][key] = {'status': backward_status, 'bits': [0] * size, 'native_bits': [''] * size}
                elif grad is None:
                    row['grad'][key] = {'status': 'GRAD_MISSING', 'bits': [0] * size, 'native_bits': [''] * size}
                else:
                    bits, raw, _ = _round_output(grad[j], dtype, method)
                    row['grad'][key] = {'status': 'OK', 'bits': bits, 'native_bits': raw}
        out.append(row)
    return out

def _rows(config, dtype, method, frozen, batch, device, diagnostic_count=0):
    n = len(frozen['sample_id'])
    from ..fp.ulp import ulp_distance
    rows = []
    batch_diffs = Counter()
    batch_max_ulp = Counter()
    for start in range(0, n, batch):
        indices = list(range(start, min(n, start + batch)))
        result = _call_batch(config, dtype, method, frozen, indices, device)
        if len(indices) > 1 and any((x['status'] in ('CALL_ERROR', 'BACKWARD_ERROR') for x in result)):
            result = [_call_batch(config, dtype, method, frozen, [i], device)[0] for i in indices]
        rows.extend(result)
    if diagnostic_count:
        indices = list(range(min(n, diagnostic_count)))
        batched = _call_batch(config, dtype, method, frozen, indices, device)
        individual = [_call_batch(config, dtype, method, frozen, [i], device)[0] for i in indices]
        for b, s in zip(batched, individual):
            if b.get('kl_bits') != s.get('kl_bits'):
                batch_diffs['KL'] += 1
                try:
                    batch_max_ulp['KL'] = max(batch_max_ulp['KL'], ulp_distance(int(b['kl_bits']), int(s['kl_bits']), dtype))
                except (ValueError, KeyError):
                    pass
            for key in b.get('grad', {}):
                if b['grad'][key]['bits'] != s.get('grad', {}).get(key, {}).get('bits'):
                    batch_diffs[f'G:{key}'] += 1
                    try:
                        differences = [ulp_distance(int(v), int(w), dtype) for v, w in zip(b['grad'][key]['bits'], s['grad'][key]['bits'])]
                        if differences:
                            metric = f'G:{key}'
                            batch_max_ulp[metric] = max(batch_max_ulp[metric], max(differences))
                    except ValueError:
                        pass
    return (rows, {'difference_count': dict(batch_diffs), 'max_ulp': dict(batch_max_ulp), 'checked': min(n, diagnostic_count), 'kl_difference_frequency': batch_diffs['KL'] / min(n, diagnostic_count) if min(n, diagnostic_count) else None})

def _pack(rows, spec, dtype, ids):
    utype = np.uint16 if dtype == 'fp16' else np.uint32
    data = {'sample_id': ids.copy(), 'status': text_array([r['status'] for r in rows]), 'error': text_array([r.get('error', '') for r in rows]), 'kl_bits': np.asarray([r.get('kl_bits', 0) for r in rows], dtype=utype), 'kl_native_bits': text_array([r.get('kl_native_bits', '') for r in rows]), 'native_dtype': text_array([r.get('native_dtype', '') for r in rows]), 'cached_p': text_array([r.get('cached_p', '') for r in rows]), 'cached_q': text_array([r.get('cached_q', '') for r in rows])}
    for side in ('P', 'Q'):
        for name, desc in spec.items():
            if not desc['differentiable']:
                continue
            key = f'{side}__{name}'
            length = int(np.prod(desc['shape']))

            def get(r, field, default):
                return r.get('grad', {}).get(key, {}).get(field, default)
            data[f'grad__{key}__bits'] = np.asarray([get(r, 'bits', [0] * length) for r in rows], dtype=utype).reshape(len(rows), length)
            data[f'grad__{key}__native_bits'] = np.asarray([get(r, 'native_bits', [''] * length) for r in rows], dtype=np.str_).reshape(len(rows), length)
            data[f'grad__{key}__status'] = text_array([get(r, 'status', r['status']) for r in rows])
    return data

def run_cell(task):
    input_path, method, batch, device, root, diagnostic_count = task
    root = Path(root)
    input_path = Path(input_path)
    frozen, meta = load_unit(input_path)
    config = meta['config']
    dtype = meta['dtype']
    dest = root / 'results' / 'raw' / input_path.relative_to(root / 'data' / 'frozen').with_name(f'{input_path.stem}__{method}.npz')
    if dest.exists() or Path(str(dest) + '.manifest.json').exists():
        from ..pipeline_io import verify
        return {'path': str(dest), 'skipped': True, 'count': verify(dest)['actual_count']}
    frozen['_parameter_shapes'] = meta['parameter_shapes']
    import torch
    torch.set_num_threads(1)
    start = time.monotonic()
    rows, diffs = _rows(config, dtype, method, frozen, batch, device, diagnostic_count)
    errors = dict(Counter((row['status'] for row in rows)))
    arrays = _pack(rows, frozen['_parameter_shapes'], dtype, frozen['sample_id'])
    details = {'stage': 'run', 'method': method, 'config': config, 'dtype': dtype, 'group': meta['group'], 'region': meta['region'], 'input_sha256': sha256(input_path), 'batch_size': batch, 'device': str(device), 'thread_count': 1, 'torch': torch.__version__, 'torch_config': torch.__config__.show(), 'validate_args': 'default', 'actual_count': len(rows), 'status_counts': errors, 'batch_invariance_checked': min(len(rows), diagnostic_count), 'batch_invariance_differences': diffs, 'elapsed_seconds': time.monotonic() - start}
    save_unit(dest, arrays, details, project_root=root)
    return {'path': str(dest), 'count': len(rows), 'seconds': details['elapsed_seconds'], 'status': errors}

def _double_rounding_diagnostic(torch, device):
    from ..fp import BINARY16, BINARY32, round_fraction
    exact = Fraction(1) + Fraction(1, 2048) + Fraction(1, 2 ** 40)
    direct = round_fraction(exact, BINARY16)
    x = torch.tensor(float(exact), dtype=torch.float64, device=device)
    native = int(x.to(dtype=torch.float16).cpu().numpy().view(np.uint16))
    via32 = int(x.to(dtype=torch.float32).to(dtype=torch.float16).cpu().numpy().view(np.uint16))
    middle32 = Fraction(1) + Fraction(1, 2 ** 24) + Fraction(1, 2 ** 50)
    direct32 = round_fraction(middle32, BINARY32)
    native32 = int(torch.tensor(float(middle32), dtype=torch.float64, device=device).to(dtype=torch.float32).cpu().numpy().view(np.uint32))
    return {'value': str(exact), 'direct': f'0x{direct:04x}', 'torch_direct': f'0x{native:04x}', 'torch_via32': f'0x{via32:04x}', 'direct_matches': native == direct, 'via32_matches': via32 == direct, 'fp32_value': str(middle32), 'fp32_direct': f'0x{direct32:08x}', 'fp32_torch': f'0x{native32:08x}', 'fp32_matches': native32 == direct32}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--configs', nargs='+', default=None)
    parser.add_argument('--dtypes', nargs='+', default=['fp16', 'fp32'])
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--root', type=Path, default=project_root())
    parser.add_argument('--diagnose-batch', type=int, default=0, help='compare up to 500 per unit')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    if args.device != 'cpu' and (not torch.cuda.is_available()):
        parser.error('CUDA unavailable')
    if args.diagnose_batch > 500:
        parser.error('--diagnose-batch maximum is 500')
    print('double_rounding', json.dumps(_double_rounding_diagnostic(torch, args.device)), flush=True)
    files = sorted((args.root / 'data' / 'frozen' / args.run_id).glob('*/*/*.npz'))
    tasks = []
    for path in files:
        parts = path.relative_to(args.root / 'data' / 'frozen' / args.run_id).parts
        config, dtype = parts[:2]
        if args.configs and config not in args.configs:
            continue
        if dtype not in args.dtypes:
            continue
        for method in ('A', 'B32', 'B64') if dtype == 'fp16' else ('A', 'B64'):
            tasks.append((str(path), method, args.batch_size, args.device, str(args.root), args.diagnose_batch))
    start = time.monotonic()
    results = []
    if args.workers == 1:
        for task in tasks:
            result = run_cell(task)
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_cell, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
    print(json.dumps({'stage': 'run', 'units': len(results), 'seconds': time.monotonic() - start}))
if __name__ == '__main__':
    main()
