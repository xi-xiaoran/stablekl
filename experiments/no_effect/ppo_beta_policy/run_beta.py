import argparse, json, math, time, torch
from tensordict import TensorDict
from torch.distributions import Beta, kl_divergence
from rsl_rl.env import VecEnv
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import BetaDistribution
LOG = []
MODE = {'m': 'official'}

def kl_wrapped(self, old_params, new_params):
    oa, ob = old_params
    na, nb = new_params
    k32 = kl_divergence(Beta(oa, ob), Beta(na, nb)).sum(dim=-1)
    k64 = kl_divergence(Beta(oa.double(), ob.double()), Beta(na.double(), nb.double())).sum(dim=-1)
    LOG.append((float(k32.mean()), float(k64.mean()), float(torch.cat([oa, ob]).max()), float(torch.cat([oa, ob]).median())))
    return k32 if MODE['m'] == 'official' else k64.float()
BetaDistribution.kl_divergence = kl_wrapped

class Reach(VecEnv):

    def __init__(self, n, A, T, device, seed, step=0.2, scale=10.0):
        self.num_envs, self.num_actions, self.max_episode_length, self.device, self.cfg = (n, A, T, device, {})
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.step_size, self.scale = (step, scale)
        self.x = torch.zeros(n, A, device=device)
        self.goal = torch.zeros(n, A, device=device)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
        self._reset(torch.ones(n, dtype=torch.bool, device=device))

    def _reset(self, m):
        k = int(m.sum())
        if k:
            self.x[m] = torch.rand(k, self.num_actions, generator=self.g, device=self.device) * 2 - 1
            self.goal[m] = torch.rand(k, self.num_actions, generator=self.g, device=self.device) * 1.6 - 0.8
            self.episode_length_buf[m] = 0

    def get_observations(self):
        return TensorDict({'policy': torch.cat([self.x, self.goal], -1)}, batch_size=[self.num_envs], device=self.device)

    def step(self, a):
        a = a.clamp(-1, 1)
        self.x = (self.x + self.step_size * a).clamp(-1, 1)
        self.episode_length_buf += 1
        r = -self.scale * ((self.x - self.goal) ** 2).mean(-1) - 0.01 * (a ** 2).mean(-1)
        done = self.episode_length_buf >= self.max_episode_length
        self._reset(done)
        return (self.get_observations(), r, done.float(), {'time_outs': done.clone()})

def rule(k, d):
    return 1 if k > 2 * d else -1 if k < d / 2 and k > 0 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='official')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--envs', type=int, default=1024)
    ap.add_argument('--actions', type=int, default=12)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--desired-kl', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--ep-len', type=int, default=32)
    ap.add_argument('--step-size', type=float, default=0.2)
    ap.add_argument('--scale', type=float, default=10.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    MODE['m'] = a.mode
    torch.manual_seed(a.seed)
    dev = 'cpu'
    torch.set_num_threads(1)
    env = Reach(a.envs, a.actions, a.ep_len, dev, a.seed, a.step_size, a.scale)
    mdl = dict(class_name='MLPModel', hidden_dims=[128, 128, 128], activation='elu', obs_normalization=False)
    cfg = dict(num_steps_per_env=a.steps, obs_groups={'actor': ['policy'], 'critic': ['policy']}, multi_gpu=None, algorithm=dict(class_name='PPO', schedule='adaptive', desired_kl=a.desired_kl, entropy_coef=a.entropy, learning_rate=a.lr), actor=dict(mdl, distribution_cfg=dict(class_name='BetaDistribution')), critic=dict(mdl))
    obs = env.get_observations()
    alg = PPO.construct_algorithm(obs, env, cfg, dev)
    alg.train_mode()
    hist = []
    t0 = time.time()
    for it in range(a.iters):
        rs = []
        with torch.inference_mode():
            for _ in range(a.steps):
                act = alg.act(obs)
                obs, r, d, ex = env.step(act)
                alg.process_env_step(obs, r, d, ex)
                rs.append(float(r.mean()))
            alg.compute_returns(obs)
        n0 = len(LOG)
        alg.update()
        calls = LOG[n0:]
        mism = sum((rule(k32, a.desired_kl) != rule(k64, a.desired_kl) for k32, k64, _, _ in calls))
        nonpos = sum((k32 <= 0 for k32, *_ in calls))
        hist.append(dict(it=it, reward=sum(rs) / len(rs), lr=alg.learning_rate, calls=len(calls), mismatch=mism, nonpos=nonpos, kl64_median=sorted((k for _, k, _, _ in calls))[len(calls) // 2], conc_max=max((c[2] for c in calls)), conc_median=calls[-1][3]))
    json.dump(dict(args=vars(a), hist=hist, seconds=time.time() - t0, log=[list(x) for x in LOG]), open(a.out, 'w'))
    h = hist
    q = lambda lo, hi, k: sum((x[k] for x in h[lo:hi])) / (hi - lo)
    n = len(h)
    print(f"{a.mode} seed {a.seed}: reward first/last 20 {q(0, 20, 'reward'):.3f} / {q(n - 20, n, 'reward'):.3f}, lr last 20 {q(n - 20, n, 'lr'):.2e}, decisions {sum((x['calls'] for x in h))}, official != exact {sum((x['mismatch'] for x in h))}, reported KL <= 0: {sum((x['nonpos'] for x in h))}, max concentration {max((x['conc_max'] for x in h)):.0f}, median conc. (end) {h[-1]['conc_median']:.1f}, {time.time() - t0:.0f} s", flush=True)
if __name__ == '__main__':
    main()
