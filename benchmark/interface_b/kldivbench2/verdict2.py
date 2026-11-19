from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
LEVEL_B = ('ULP8', 'REL1', 'REL10', 'REL100', 'REL_L2_1', 'COS_999', 'COS_99', 'COS_9')
LEVEL_C = ('NAN', 'FALSE_INF', 'MISSED_INF', 'SIGN_MISMATCH', 'FALSE_ZERO', 'NONZERO_AT_EXACT_ZERO', 'SIGN', 'VANISH', 'SPURIOUS', 'REVERSED', 'VANISH_VEC')
S14 = ('S1', 'S1t', 'S1s', 'S1ts', 'S2', 'S2b', 'S3', 'S4')

def side_of(metric):
    h = metric.split('_', 1)[0]
    return 'fwd' if h in ('E', 'S') else 'bwd' if h in ('G', 'GV') else None

def kind_of(metric):
    m = metric.split(':')[0].split('_', 1)[1] if '_' in metric else ''
    return 'B' if m in LEVEL_B else 'C' if m in LEVEL_C else None

def collect(cfgdir):
    sc = defaultdict(Counter)
    pr = defaultdict(Counter)
    h1 = defaultdict(Counter)
    ev = Counter()
    hb = defaultdict(Counter)
    s14 = defaultdict(Counter)
    ex = defaultdict(list)
    nf = 0
    methods = set()
    for f in sorted((cfgdir / 'score').glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        nf += 1
        for key, r in d['results'].items():
            m = key.split('|')[1]
            if m.startswith('PAIR:'):
                pr[m[5:]].update(r['cnt'])
            else:
                sc[m].update({k: v for k, v in r['cnt'].items() if 'maxULP' not in k})
                for k, v in r['cnt'].items():
                    if 'maxULP' in k:
                        sc[m][k] = max(sc[m][k], v)
                if d['layer'] in S14:
                    s14[m].update({k: v for k, v in r['cnt'].items() if k.startswith('S_ULP8:')})
                if m != 'A':
                    methods.add(m)
        h = json.loads((cfgdir / 'h1' / f.name).read_text(encoding='utf-8'))['h1']
        for B, hbk in h.items():
            ev.update(hbk['evec'])
            for m, hm in hbk['methods'].items():
                h1[m].update(hm['cnt'])
                h1[m]['n_regressions'] += hm['n_regressions']
                ex[m] += hm['new_bad_examples'][:3]
        g = cfgdir / 'h1b' / f.name
        if g.exists():
            for key, r in json.loads(g.read_text(encoding='utf-8'))['h1b'].items():
                m = key.split('|')[1]
                for k, v in r['cnt'].items():
                    if k.endswith('max_ulp'):
                        hb[m][k] = max(hb[m][k], v)
                    else:
                        hb[m][k] += v
                hb[m]['files'] += 1
    return dict(files=nf, methods=sorted(methods), score=sc, pairs=pr, h1=h1, evec=ev, h1b=hb, s14=s14, examples=ex)

def judge(c, m, lt):
    sc, pr, h1, ev, hb = (c['score'][m], c['pairs'][m], c['h1'][m], c['evec'], c['h1b'][m])
    out = {}
    for side in ('fwd', 'bwd'):
        cls = ('E', 'S:') if side == 'fwd' else ('G:', 'BG:')
        isc = lambda k: k.split('|')[0].startswith(cls) if side == 'bwd' else k.split('|')[0] == 'E' or k.startswith('S:')
        tot = Counter()
        for k, v in h1.items():
            if '|' in k and isc(k) and (not k.startswith('H1a')):
                tot[k.split('|')[1]] += v
        hbt = Counter()
        for k, v in hb.items():
            if '|' in k and isc(k):
                suf = k.split('|')[1]
                if suf == 'max_ulp':
                    hbt[suf] = max(hbt[suf], v)
                else:
                    hbt[suf] += v
        nb_sample = {k.rsplit('|', 1)[0]: v for k, v in pr.items() if k.endswith('|new_bad') and v and (side_of(k) == side) and kind_of(k.rsplit('|', 1)[0])}
        nb_comp = {k.split('|', 1)[1]: v for k, v in h1.items() if k.startswith('H1a_new_bad|') and v and (side_of(k.split('|', 1)[1]) == side)}
        bad = lambda pat: sum((v for k, v in sc.items() if k.endswith('|bad') and k.split('|')[0].split(':')[0] in pat))
        h2 = {}
        if side == 'fwd':
            h2['NaN where the truth is defined'] = (bad(('E_NAN', 'S_NAN')), True)
            h2['false inf'] = (bad(('E_FALSE_INF', 'S_FALSE_INF')), True)
            h2['false zero'] = (bad(('E_FALSE_ZERO', 'S_FALSE_ZERO')), m != 'C')
            h2['reductions >= 8 ULP (S1-S4)'] = (sum((v for k, v in c['s14'][m].items() if k.endswith('|bad'))), m == 'D64')
        else:
            h2['NaN where the truth is defined'] = (bad(('G_NAN',)), True)
            h2['false inf'] = (bad(('G_FALSE_INF',)), True)
            bn = sum((v for k, v in sc.items() if k.startswith('BG:') and k.endswith('|n')))
            bm = sum((v for k, v in sc.items() if k.startswith('BG:') and k.endswith('|BG_MATCH')))
            h2['boundary gradients not matching'] = (bn - bm, True)
            req = not (m == 'C' and lt)
            for nm in ('EIMPL_REL1', 'EIMPL_SIGN', 'EIMPL_ZERO'):
                h2[f'{nm} (against the certified RN gradient)'] = (sum((v for k, v in ev.items() if k.startswith(f'{m}|{nm}') and k.endswith('|bad'))), req)
            h2['E_impl undecided'] = (sum((v for k, v in ev.items() if k.startswith(f'{m}|EIMPL') and k.endswith('|und'))), True)
            h2['rows without gradient output'] = (sum((v for k, v in ev.items() if k.startswith(f'{m}|EIMPL_NO_OUTPUT'))), True)
        unchecked = tot['regressed'] - hbt['regressed']
        hard = bool(nb_sample or nb_comp or hbt['violation'] or hbt['undecided'] or unchecked or any((v and req for v, req in h2.values())))
        tier = 'NOT PASSED' if hard else 'FULL FIX' if tot['regressed'] == 0 else 'PARTIAL FIX'
        out[side] = dict(tier=tier, components=tot['n'], cr_A=tot['crA'], cr_M=tot['crM'], fixed=tot['fixed'], regressed=tot['regressed'], both_not_cr=tot['both_not_cr'], regress_checked=hbt['regressed'], exception=hbt['exception'], violation=hbt['violation'], undecided=hbt['undecided'], unchecked=unchecked, max_ulp_regression=hbt['max_ulp'], H1a_new_bad_sample=nb_sample, H1a_new_bad_component=nb_comp, H2={k: dict(events=v, required=r) for k, (v, r) in h2.items()})
    return out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root2', required=True)
    ap.add_argument('--cfg', default='*')
    ap.add_argument('--out', default=None)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--formal', default='D64', help='the formal fix: with --strict any NOT PASSED of this method is a failure (other methods are exploratory and only recorded)')
    a = ap.parse_args(argv)
    root2 = Path(a.root2)
    out = Path(a.out or root2 / 'verdict')
    out.mkdir(parents=True, exist_ok=True)
    allv = {}
    L = []
    for cd in sorted((p for p in root2.glob(a.cfg) if (p / 'score').is_dir())):
        c = collect(cd)
        lt = cd.name.startswith('ltT')
        v = {m: judge(c, m, lt) for m in c['methods']}
        allv[cd.name] = v
        L.append(f"## {cd.name}  ({c['files']} units)\n")
        L.append('| method | side | verdict | components | CR in A | CR in method | fixed | regressed | exception | violation | undecided | unchecked | max ULP of a regression |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for m in c['methods']:
            for side in ('fwd', 'bwd'):
                j = v[m][side]
                L.append(f"| {m} | {side} | **{j['tier']}** | {j['components']} | {j['cr_A']} | {j['cr_M']} | {j['fixed']} | {j['regressed']} | {j['exception']} | {j['violation']} | {j['undecided']} | {j['unchecked']} | {j['max_ulp_regression']} |")
        L.append('\n| method | side | H2 item | events | required |\n|---|---|---|---|---|')
        for m in c['methods']:
            for side in ('fwd', 'bwd'):
                for k, d in v[m][side]['H2'].items():
                    L.append(f"| {m} | {side} | {k} | {d['events']} | {('yes' if d['required'] else 'report')} |")
        L.append('\n| method | side | H1a new_bad (sample level, frozen pairing) | H1a new_bad (component level) |\n|---|---|---|---|')
        for m in c['methods']:
            for side in ('fwd', 'bwd'):
                L.append(f"| {m} | {side} | {v[m][side]['H1a_new_bad_sample'] or 0} | {v[m][side]['H1a_new_bad_component'] or 0} |")
        L.append('\n| method | class | components | bits differ from A | fixed | regressed | both not CR: closer / farther / same | A not finite there | max ULP of the method |\n|---|---|---|---|---|---|---|---|---|')
        for m in c['methods']:
            h = c['h1'][m]
            agg = defaultdict(Counter)
            for k, x in h.items():
                if '|' in k and (not k.startswith('H1a')):
                    cl = k.split('|')[0]
                    g = 'E' if cl == 'E' else cl.split(':')[0] + ':' + (cl.split(':')[1] if cl[0] in 'GB' else '')
                    agg[g][k.split('|')[1]] += x
            mx = defaultdict(int)
            for k, x in c['score'][m].items():
                if 'maxULP' in k:
                    g = 'E' if k.startswith('E_') else 'S:' if k.startswith('S_') else 'G:' + k.split(':')[1]
                    mx[g] = max(mx[g], x)
            for g in sorted(agg):
                a_ = agg[g]
                L.append(f"| {m} | {g} | {a_['n']} | {a_['differ']} | {a_['fixed']} | {a_['regressed']} | {a_['bnc_closer']} / {a_['bnc_farther']} / {a_['bnc_same_distance']} | {a_['bnc_A_nonfinite']} | {mx.get(g, '')} |")
        ev = c['evec']
        tot = lambda pre: (sum((x for k, x in ev.items() if k.startswith(pre) and k.endswith('|bad'))), sum((x for k, x in ev.items() if k.startswith(pre) and k.endswith('|n'))))
        L.append('\n| quantity | rows >= 1% | rows |\n|---|---|---|')
        b, n = tot('EFORMAT_REL1')
        L.append(f'| E_format | {b} | {n} |')
        for m in ['A'] + c['methods']:
            b, n = tot(f'{m}|EIMPL_REL1')
            L.append(f'| E_impl {m} | {b} | {n} |')
        L.append('')
        (out / f'examples_{cd.name}.json').write_text(json.dumps({m: c['examples'][m][:20] for m in c['methods']}, indent=1), encoding='utf-8')
    failed = [f'{cfg}:{a.formal}:{side}' for cfg, v in allv.items() for side in ('fwd', 'bwd') if a.formal not in v or v[a.formal][side]['tier'] == 'NOT PASSED']
    explor = [f'{cfg}:{m}:{side}' for cfg, v in allv.items() for m in v if m != a.formal for side in ('fwd', 'bwd') if v[m][side]['tier'] == 'NOT PASSED']
    status = dict(formal_method=a.formal, formal_pass=bool(allv) and (not failed), formal_not_passed=failed, exploratory_not_passed=explor)
    L.insert(0, f"**Formal fix {a.formal}: {('PASS' if status['formal_pass'] else 'NOT PASSED ' + str(failed))}.**  Exploratory methods NOT PASSED: {explor or 'none'}\n")
    (out / 'verdict.json').write_text(json.dumps(allv, indent=1), encoding='utf-8')
    (out / 'status.json').write_text(json.dumps(status, indent=1), encoding='utf-8')
    (out / 'VERDICT.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    if a.strict and (not allv):
        sys.exit(5)
    if a.strict and failed:
        print(f'STRICT FAILURE: formal method {a.formal} NOT PASSED: {failed}', file=sys.stderr)
        sys.exit(9)
if __name__ == '__main__':
    main()
