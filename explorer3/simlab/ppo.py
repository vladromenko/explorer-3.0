"""Asymmetric PPO on batched GPU environments (M6, line S).

Actor: contract observation only (what the real robot has) + a short history.
Critic: privileged simulator state (true pose error, velocities, drive parameters).
This is the teacher-student idea in one stage: the actor never sees privileged data, so
no separate distillation is needed for the base tasks; DAgger remains available if the
student lags the critic-informed returns.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class History:
    """Stacks the last k actor observations (oldest first). Same logic in policy_runtime."""

    def __init__(self, n, dim, k, device):
        self.buf = torch.zeros(n, k, dim, device=device)

    def push(self, obs, reset_mask=None):
        if reset_mask is not None and reset_mask.any():
            self.buf[reset_mask] = obs[reset_mask, None].expand(-1, self.buf.shape[1], -1)
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.buf[:, -1] = obs
        return self.buf.flatten(1)


def mlp(i, o, hidden=(256, 256)):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))
        self.frozen = False

    def update(self, x):
        if self.frozen:
            return
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class ActorCritic(nn.Module):
    def __init__(self, actor_dim, critic_dim, act_dim):
        super().__init__()
        self.actor_norm, self.critic_norm = RunningNorm(actor_dim), RunningNorm(critic_dim)
        self.actor = mlp(actor_dim, act_dim)
        self.critic = mlp(critic_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, a_obs):
        mu = self.actor(self.actor_norm(a_obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, c_obs):
        return self.critic(self.critic_norm(c_obs)).squeeze(-1)


def train(make_env, *, robot, out_dir, total_steps=50_000_000, n=4096, horizon=32, history=3, epochs=5,
          minibatches=4, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.003, max_minutes=None, seed=0,
          curriculum_up=0.7, log=print):
    torch.manual_seed(seed)
    env = make_env(n=n, seed=seed, curriculum=0.0)
    obs, priv = env.reset_all()
    dev = obs.device
    hist = History(n, obs.shape[1], history, dev)
    a_obs = hist.push(obs, torch.ones(n, dtype=torch.bool, device=dev))
    model = ActorCritic(a_obs.shape[1], priv.shape[1], env.adim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = math.ceil(total_steps / (n * horizon))
    start = time.time()
    window = {'success': [], 'collision': [], 'timeout': []}
    best = -1.0
    history_log = []
    for it in range(iters):
        buf = {k: [] for k in ('a_obs', 'priv', 'act', 'logp', 'val', 'rew', 'done')}
        ep_succ = ep_coll = ep_to = 0
        with torch.no_grad():
            for t in range(horizon):
                model.actor_norm.update(a_obs)
                model.critic_norm.update(priv)
                d = model.dist(a_obs)
                act = d.sample()
                buf['a_obs'].append(a_obs); buf['priv'].append(priv); buf['act'].append(act)
                buf['logp'].append(d.log_prob(act).sum(-1)); buf['val'].append(model.value(priv))
                obs, priv, rew, term, trunc, info = env.step(act)
                done = term | trunc
                a_obs = hist.push(obs, done)
                buf['rew'].append(rew); buf['done'].append(done.float())
                ep_succ += int(info['success'].sum()); ep_coll += int(info['collision'].sum()); ep_to += int(info['timeout'].sum())
            last_val = model.value(priv)
            S = {k: torch.stack(v) for k, v in buf.items()}
            adv = torch.zeros_like(S['rew'])
            gae = torch.zeros(n, device=dev)
            for t in reversed(range(horizon)):
                nv = last_val if t == horizon - 1 else S['val'][t + 1]
                nonterm = 1.0 - S['done'][t]
                delta = S['rew'][t] + gamma * nv * nonterm - S['val'][t]
                gae = delta + gamma * lam * nonterm * gae
                adv[t] = gae
            ret = adv + S['val']
        flat = lambda x: x.reshape(horizon * n, *x.shape[2:])
        A, P, ACT, LP, ADV, RET = map(flat, (S['a_obs'], S['priv'], S['act'], S['logp'], adv, ret))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        size = horizon * n // minibatches
        for _ in range(epochs):
            perm = torch.randperm(horizon * n, device=dev)
            for j in range(minibatches):
                idx = perm[j * size:(j + 1) * size]
                d = model.dist(A[idx])
                lp = d.log_prob(ACT[idx]).sum(-1)
                ratio = (lp - LP[idx]).exp()
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - clip, 1 + clip) * ADV[idx]).mean()
                vl = 0.5 * ((model.value(P[idx]) - RET[idx]) ** 2).mean()
                loss = pg + vl - ent * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        episodes = max(1, ep_succ + ep_coll + ep_to)
        for k, v in (('success', ep_succ), ('collision', ep_coll), ('timeout', ep_to)):
            window[k] = (window[k] + [v / episodes])[-10:]
        rate = float(np.mean(window['success']))
        if rate > curriculum_up and env.curriculum < 1.0 and len(window['success']) == 10:
            env.curriculum = min(1.0, env.curriculum + 0.1)
            window = {k: [] for k in window}
        steps = (it + 1) * n * horizon
        row = {'iter': it, 'steps': steps, 'minutes': (time.time() - start) / 60, 'curriculum': env.curriculum,
               'success': rate, 'collision': float(np.mean(window['collision'] or [0])),
               'timeout': float(np.mean(window['timeout'] or [0])), 'episodes': episodes,
               'std': float(model.log_std.exp().mean())}
        history_log.append(row)
        if it % 10 == 0:
            log(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
        if env.curriculum >= 1.0 and rate >= best:
            best = rate
            torch.save(model.state_dict(), out_dir / 'best.pt')
        if max_minutes and (time.time() - start) / 60 > max_minutes:
            log(f'stopping at time budget {max_minutes} min')
            break
    torch.save(model.state_dict(), out_dir / 'last.pt')
    (out_dir / 'train_log.json').write_text(json.dumps(history_log))
    meta = {'robot': robot, 'history': history, 'actor_dim': a_obs.shape[1], 'critic_dim': priv.shape[1],
            'act_dim': env.adim, 'final_curriculum': env.curriculum, 'best_full_curriculum_success': best,
            'steps': steps, 'minutes': (time.time() - start) / 60, 'seed': seed}
    (out_dir / 'train_meta.json').write_text(json.dumps(meta, indent=1))
    return model, meta
