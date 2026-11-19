import csv
import numpy as np
from _paths import RESULTS
BINS = ['B_c'] + [f'B_{i}' for i in range(1, 15)]
MV = ('categorical', 'contbern', 'dirichlet', 'mvn', 'lowrank')

def _read(name):
    with open(RESULTS / 'heatmaps' / name, newline='') as f:
        return list(csv.DictReader(f))
_A = {}
for r in _read('interface_a_kl_rel1_r0.csv'):
    _A[r['config'], r['dtype'], r['kl_bin'], r['method'], r['device']] = (int(r['n']), int(r['events']))
_B = {}
for r in _read('interface_b_s_rel1_r0.csv'):
    _B[r['config'], r['kl_bin'], r['method'], r['device']] = (int(r['n']), int(r['events']))
_cfgs = sorted({k[0] for k in _A})
SC = sorted(c for c in _cfgs if not c.startswith(MV))
ROWSA = SC + sorted(c for c in _cfgs if c not in SC)

def matA(dt, method, device, nmin=100):
    M = np.full((len(ROWSA), 15), np.nan)
    for i, c in enumerate(ROWSA):
        for j, b in enumerate(BINS):
            n, e = _A.get((c, dt, b, method, device), (0, 0))
            if n >= nmin:
                M[i, j] = 100 * e / n
    return M

def cfgsB(dt):
    return [f'lt{l}_{dt}_K{K}' for l in 'FT' for K in (2, 4, 16, 256, 1024)]

def rowsB(dt):
    return [f"{('log_target' if c[2] == 'T' else 'probability target')}, K={c.split('K')[1]}" for c in cfgsB(dt)]

def matB(dt, method, device, nmin=100):
    cf = cfgsB(dt)
    M = np.full((len(cf), 15), np.nan)
    for i, c in enumerate(cf):
        for j, b in enumerate(BINS):
            n, e = _B.get((c, b, method, device), (0, 0))
            if n >= nmin:
                M[i, j] = 100 * e / n
    return M
