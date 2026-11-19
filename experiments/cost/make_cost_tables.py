import json, sys
src, out = (sys.argv[1], sys.argv[2])
G = json.load(open(f'{src}/cost_cuda.json'))
C = json.load(open(f'{src}/cost_cpu.json'))
f = lambda x: f'{x:.0f}' if x >= 9.95 else f'{x:.1f}'

def closed(R, name, n, k):
    r = next((x for x in R['closed'] if x['name'] == name and x['n'] == n))
    return r[k] / r['official_ms']
rows = []
for name in [x['name'] for x in G['closed'] if x['n'] == 65536]:
    cell = lambda R, k: f'{f(closed(R, name, 65536, k))} / {f(closed(R, name, 1048576, k))}'
    rows.append(f"{name} & {cell(G, 'official_fp64_ms')} & {cell(G, 'rewrite_ms')} & {cell(G, 'ours_ms')} & {cell(C, 'official_fp64_ms')} & {cell(C, 'rewrite_ms')} & {cell(C, 'ours_ms')}\\\\")
open(f'{out}/tab_cost_closed_rows.tex', 'w').write('\n'.join(rows) + '\n\\bottomrule\n')
rows = []
for g in G['kldiv']:
    c = next((x for x in C['kldiv'] if (x['dtype'], x['log_target'], x['V']) == (g['dtype'], g['log_target'], g['V'])))
    r = lambda x, k: f(x[k] / x['official_ms'])
    rows.append(f"{g['dtype'].upper()} & {('True' if g['log_target'] else 'False')} & ${g['B']}\\times{g['V']:,}$".replace(',', '{,}') + f" & {r(g, 'official_fp64_ms')} & {r(g, 'repaired_nocheck_ms')} & {r(g, 'repaired_ms')} & {r(c, 'official_fp64_ms')} & {r(c, 'repaired_nocheck_ms')} & {r(c, 'repaired_ms')}\\\\")
open(f'{out}/tab_cost_b_rows.tex', 'w').write('\n'.join(rows) + '\n\\bottomrule\n')
rows = []
for g in G['logits']:
    c = next((x for x in C['logits'] if (x['dtype'], x['B'], x['V']) == (g['dtype'], g['B'], g['V'])), None)
    r = lambda x, k: '--' if x is None else f(x[k] / x['official_ms'])
    rows.append(f"{g['dtype'].upper()} & ${g['B']}\\times{g['V']:,}$".replace(',', '{,}') + f" & {g['official_ms']:.2f} & {r(g, 'official_fp64_ms')} & {r(g, 'logits_path_ms')} & {r(c, 'official_fp64_ms')} & {r(c, 'logits_path_ms')}\\\\")
open(f'{out}/tab_cost_logits_rows.tex', 'w').write('\n'.join(rows) + '\n\\bottomrule\n')
five = ['Exponential', 'Laplace', 'Poisson', 'Normal', 'Bernoulli (logits)']
rng = lambda R, names, k: (min((closed(R, n, N, k) for n in names for N in (65536, 1048576))), max((closed(R, n, N, k) for n in names for N in (65536, 1048576))))
for dev, R in (('GPU', G), ('CPU', C)):
    print(dev, 'five closed forms: FP64 %.1f-%.1f, Rewrite %.1f-%.1f, Ours %.1f-%.1f' % (*rng(R, five, 'official_fp64_ms'), *rng(R, five, 'rewrite_ms'), *rng(R, five, 'ours_ms')), '| Gamma Rewrite %.1f-%.1f Ours %.1f-%.1f | Beta Rewrite %.1f-%.1f Ours %.0f-%.0f' % (*rng(R, ['Gamma'], 'rewrite_ms'), *rng(R, ['Gamma'], 'ours_ms'), *rng(R, ['Beta'], 'rewrite_ms'), *rng(R, ['Beta'], 'ours_ms')))
    k = [(x['repaired_ms'] / x['official_ms'], x['repaired_nocheck_ms'] / x['repaired_ms']) for x in R['kldiv']]
    print('   kl_div repaired %.0f-%.0f, check share %.0f%%-%.0f%%' % (min((a for a, _ in k)), max((a for a, _ in k)), 100 * (1 - max((b for _, b in k))), 100 * (1 - min((b for _, b in k)))), '| logits path %.1f-%.1f, official FP64 %.1f-%.1f, path / FP64 %.1f-%.1f' % (min((x['logits_path_ms'] / x['official_ms'] for x in R['logits'])), max((x['logits_path_ms'] / x['official_ms'] for x in R['logits'])), min((x['official_fp64_ms'] / x['official_ms'] for x in R['logits'])), max((x['official_fp64_ms'] / x['official_ms'] for x in R['logits'])), min((x['logits_path_ms'] / x['official_fp64_ms'] for x in R['logits'])), max((x['logits_path_ms'] / x['official_fp64_ms'] for x in R['logits']))))
g = next((x for x in G['logits'] if x['dtype'] == 'bf16' and x['B'] == 512))
print('GPU bf16 512x151936: official %.2f ms, logits path %.1f ms' % (g['official_ms'], g['logits_path_ms']))
