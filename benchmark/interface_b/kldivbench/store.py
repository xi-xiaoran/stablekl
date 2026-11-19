from __future__ import annotations
import hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np
SCHEMA = 'kldiv_v1'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()

def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))}

def mpath(path):
    return Path(str(path) + '.manifest.json')

def exists(path):
    return Path(path).is_file() and mpath(path).is_file()

def save(path, arrays, details):
    path = Path(path)
    if path.exists() or mpath(path).exists():
        verify(path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = f'{os.getpid()}-{os.urandom(4).hex()}'
    tmp = path.with_name(path.stem + f'.{tok}.partial.npz')
    tmpm = Path(str(mpath(path)) + f'.{tok}.partial')
    try:
        np.savez_compressed(tmp, **arrays)
        meta = dict(details, schema=SCHEMA, sha256=sha256(tmp), source_sha256=source_hashes(), python=sys.version, platform=platform.platform(), numpy=np.__version__)
        tmpm.write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True, default=str), encoding='utf-8')
        os.rename(tmp, path)
        os.rename(tmpm, mpath(path))
    finally:
        tmp.unlink(missing_ok=True)
        tmpm.unlink(missing_ok=True)
    return True

def verify(path):
    path = Path(path)
    if not exists(path):
        raise FileNotFoundError(f'incomplete unit: {path}')
    meta = json.loads(mpath(path).read_text(encoding='utf-8'))
    if meta['sha256'] != sha256(path):
        raise ValueError(f'SHA256 mismatch: {path}')
    return meta

def load(path):
    meta = verify(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return (arrays, meta)

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, path)

def cfg_name(log_target, dt, K):
    return f"lt{('T' if log_target else 'F')}_{dt}_K{K}"

def xt_hash(x, t):
    return hashlib.sha256(repr((tuple((int(v) for v in x)), tuple((int(v) for v in t)))).encode()).hexdigest()[:16]
