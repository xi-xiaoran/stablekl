from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()

def source_hashes(project_root: Path) -> dict[str, str]:
    return {str(p.relative_to(project_root)).replace('\\', '/'): sha256(p) for p in sorted((project_root / 'src' / 'klbench').rglob('*.py'))}

def text_array(values: list[Any]) -> np.ndarray:
    return np.asarray(['' if v is None else str(v) for v in values], dtype=np.str_)

def fraction_text(value: Any) -> str:
    return '' if value is None else str(value)

def manifest_path(path: Path) -> Path:
    return Path(str(path) + '.manifest.json')

def verify(path: Path) -> dict[str, Any]:
    m = manifest_path(path)
    if not path.is_file() or not m.is_file():
        raise FileNotFoundError(f'Incomplete unit: {path}')
    data = json.loads(m.read_text(encoding='utf-8'))
    if data.get('sha256') != sha256(path):
        raise ValueError(f'NPZ SHA256 mismatch: {path}')
    return data

def save_unit(path: Path, arrays: dict[str, np.ndarray], details: dict[str, Any], *, project_root: Path) -> bool:
    manifest = manifest_path(path)
    if path.exists() or manifest.exists():
        verify(path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f'{os.getpid()}-{os.urandom(5).hex()}'
    tmp_npz = path.with_name(path.stem + f'.{token}.partial.npz')
    tmp_meta = manifest.with_name(manifest.name + f'.{token}.partial')
    try:
        np.savez_compressed(tmp_npz, **arrays)
        metadata = {**details, 'schema_version': 'klbench_v1', 'sha256': sha256(tmp_npz), 'source_sha256': source_hashes(project_root), 'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__}
        tmp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding='utf-8')
        os.rename(tmp_npz, path)
        os.rename(tmp_meta, manifest)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
    return True

def load_unit(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = verify(path)
    with np.load(path, allow_pickle=False) as content:
        arrays = {name: content[name].copy() for name in content.files}
    return (arrays, manifest)

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
