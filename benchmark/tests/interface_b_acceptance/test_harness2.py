import json, os, shutil, subprocess, sys
from pathlib import Path
work = Path(sys.argv[1])
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
root = work / 'frozen'
fails = 0
checks = 0
A = ['--lt', '0', '--dt', 'fp32', '--K', '2']
env0 = dict(os.environ)

def sh(mod, stage, extra, forbid, env=None):
    e = dict(env0, KLDIV_FORBID=forbid, **env or {})
    r = subprocess.run([sys.executable, '-W', 'ignore', '-m', mod, stage] + extra + A, env=e, capture_output=True, text=True)
    return (r.returncode, r.stdout + r.stderr)

def ok(c, msg):
    global fails, checks
    checks += 1
    if not c:
        fails += 1
        print('FAIL:', msg)
for st, fb in (('gen', 'torch'), ('fill', 'flint,mpmath'), ('certify', 'torch'), ('run', 'flint,mpmath')):
    rc, out = sh('kldivbench.pipeline', st, ['--root', str(root)] + (['--quota', 'smoke'] if st == 'gen' else []), fb)
    ok(rc == 0, f'frozen stage {st}: {out[-300:]}')

def run_stages(name, inject=None):
    r2 = work / name
    env = {'KLDIV2_INJECT': inject} if inject else {}
    for st, fb in (('run2', 'flint'), ('score2', 'torch'), ('pre2', 'flint'), ('h1b', 'torch')):
        rc, out = sh('kldivbench2.pipeline2', st, ['--root', str(root), '--root2', str(r2), '--strict'], fb, env)
        if st == 'pre2' and inject:
            continue
        ok(rc == 0, f'{name} stage {st}: {out[-400:]}')
    subprocess.run([sys.executable, '-m', 'kldivbench2.verdict2', '--root2', str(r2)], env=env0, capture_output=True, text=True)
    return json.loads((r2 / 'verdict' / 'verdict.json').read_text())['ltF_fp32_K2']
v = run_stages('clean')
d = v['D64']
ok(d['fwd']['tier'] != 'NOT PASSED' and d['bwd']['tier'] != 'NOT PASSED', f"clean run: D64 accepted ({d['fwd']['tier']} / {d['bwd']['tier']})")
rc, out = sh('kldivbench2.pipeline2', 'replay2', ['--root', str(root), '--root2', str(work / 'clean'), '--strict'], 'flint')
ok(rc == 0, 'clean replay2 is bit-identical')
rc, out = sh('kldivbench2.pipeline2', 'replay2', ['--root', str(root), '--root2', str(work / 'clean'), '--strict'], 'flint', {'KLDIV_INJECT_REPLAY_DIFF': '1'})
ok(rc == 4, 'replay2 --strict fails on a differing bit')
d = run_stages('bump3', 'bump3')['D64']['fwd']
ok(d['tier'] == 'NOT PASSED' and d['violation'] > 0, f"3-ULP error in a reduction: regression counted as violation ({d['tier']}, violation {d['violation']}, exception {d['exception']})")
d = run_stages('nan_gt', 'nan_gt')['D64']['bwd']
ok(d['tier'] == 'NOT PASSED' and d['H2']['NaN where the truth is defined']['events'] > 0 and d['H1a_new_bad_component'], f"NaN gradient: H2 and H1a ({d['tier']})")
d = run_stages('scale_gt', 'scale_gt')['D64']['bwd']
ok(d['tier'] == 'NOT PASSED' and d['H2']['EIMPL_REL1 (against the certified RN gradient)']['events'] > 0, f"gradient scaled by 1.5: E_impl ({d['tier']})")
d = run_stages('false_inf', 'false_inf')['D64']['fwd']
ok(d['tier'] == 'NOT PASSED' and d['H2']['false inf']['events'] > 0, f"false inf: H2 ({d['tier']})")

def gate(name):
    r = subprocess.run([sys.executable, '-m', 'kldivbench2.verdict2', '--root2', str(work / name), '--strict'], env=env0, capture_output=True, text=True)
    return (r.returncode, json.loads((work / name / 'verdict' / 'status.json').read_text()))
rc, st = gate('clean')
ok(rc == 0 and st['formal_pass'], f'verdict --strict accepts the clean run (rc {rc})')
for nm in ('bump3', 'nan_gt', 'scale_gt', 'false_inf'):
    rc, st = gate(nm)
    ok(rc == 9 and (not st['formal_pass']) and st['formal_not_passed'], f"verdict --strict rejects '{nm}' with exit code 9 (rc {rc})")
r = subprocess.run([sys.executable, '-m', 'kldivbench2.verdict2', '--root2', str(work / 'clean'), '--strict', '--formal', 'D32'], env=env0, capture_output=True, text=True)
ok(r.returncode == 9, 'verdict --strict fails when the formal method has no results')
rc, out = sh('kldivbench2.pipeline2', 'run2', ['--root', str(root), '--root2', str(work / 'uncert'), '--strict'], 'flint', {'KLDIV2_INJECT': 'uncertified'})
lg = json.loads(sorted((work / 'uncert' / 'ltF_fp32_K2' / 'logs').glob('run2*.json'))[0].read_text())
ok(rc == 4 and lg['result']['uncertified_rounding'] > 0 and ('run2 uncertified_rounding' in lg['strict_problems']), f'an uncertified rounding fails run2 --strict (rc {rc})')
import glob
r2 = work / 'bump3'
hf = sorted(glob.glob(str(r2 / 'ltF_fp32_K2' / 'h1b' / '*.json')))
cut = 0
for f in hf:
    d = json.loads(Path(f).read_text())
    for k, v in d['h1b'].items():
        if k.endswith('|D64'):
            for c in list(v['cnt']):
                if c.startswith('S:') and c.split('|')[1] in ('regressed', 'violation', 'exception'):
                    cut += v['cnt'].pop(c)
    Path(f).write_text(json.dumps(d))
subprocess.run([sys.executable, '-m', 'kldivbench2.verdict2', '--root2', str(r2)], env=env0, capture_output=True, text=True)
d = json.loads((r2 / 'verdict' / 'verdict.json').read_text())['ltF_fp32_K2']['D64']['fwd']
ok(cut > 0 and d['tier'] == 'NOT PASSED' and (d['unchecked'] > 0) and (d['violation'] == 0), f"regressions missing from the checked list keep the side NOT PASSED (unchecked {d['unchecked']})")
print(f'TOTAL {checks} checks, {fails} failed')
sys.exit(1 if fails else 0)
