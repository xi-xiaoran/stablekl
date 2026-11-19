"""Locate the source file behind a module name and return its MD5 (recorded in run metadata)."""
import hashlib
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
_PACKAGE = {'logits_kl.py': 'stablekl.logits', 'kldivfix.py': 'stablekl.interface_b'}


def _locate(name, here):
    if name in _PACKAGE:
        import importlib
        return Path(importlib.import_module(_PACKAGE[name]).__file__)
    if name.startswith('fix_'):
        import stablekl
        return Path(stablekl.__file__).parent / '_closed' / name[4:]
    local = Path(here) / name
    if local.exists():
        return local
    hits = sorted(EXPERIMENTS.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def code_md5(name, here):
    return hashlib.md5(_locate(name, here).read_bytes()).hexdigest()
