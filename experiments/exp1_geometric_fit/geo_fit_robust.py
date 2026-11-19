import torch, math
import geo_fit as G
cases = [(torch.float16, 'probs', 0.01), (torch.float16, 'probs', 0.001), (torch.float16, 'logits', 0.01), (torch.float32, 'probs', 0.001), (torch.float32, 'logits', 0.01)]

def run(method, param, pstar, dt, opt_name, lr, p0, steps):
    th_star = math.log(pstar) - math.log1p(-pstar)
    star = float(torch.tensor(pstar if param == 'probs' else th_star, dtype=dt))
    p_opt = star if param == 'probs' else float(G.sig(star))
    th = torch.tensor([math.log(p0) - math.log1p(-p0)], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr) if opt_name == 'adam' else torch.optim.SGD([th], lr=lr, momentum=0.9)
    for s in range(steps):
        L = G.loss(method, param, 'rev', th, star, dt)
        opt.zero_grad()
        L.backward()
        opt.step()
    pt = float(torch.sigmoid(th.detach().to(dt)).double()) if param == 'probs' else float(G.sig(float(th.detach().to(dt))))
    return ((pt - p_opt) / p_opt, G.true_kl(pt, p_opt))
for dt, param, pstar in cases:
    for opt_name, lr, steps in (('adam', 0.001, 10000), ('sgd', 0.1, 10000)):
        for start in ('below', 'above', 'at'):
            p0 = {'below': pstar / 3, 'above': 3 * pstar, 'at': pstar}[start]
            out = []
            for m in ('official', 'D'):
                e, kl = run(m, param, pstar, dt, opt_name, lr, p0, steps)
                out.append(f'{m} signed rel err {e:+.2e} KL {kl:.1e}')
            print(f'{str(dt)[-7:]} {param:6s} p*={pstar:<6g} {opt_name} lr {lr:g} start {start:5s} | ' + ' | '.join(out), flush=True)
