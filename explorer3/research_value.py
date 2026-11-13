"""Value of FIXED help rules on held-out randomized episodes (inverse propensity weighting).

Brier/calibration only say whether success probabilities are well predicted. The question of the study is different:
does choosing WHEN to help by G beat simple rules? For a deterministic rule pi(x) in {0,1} and randomized assignment a with
known propensity e, the IPW estimate of success if the rule had been followed is
    V(pi) = mean_i [ 1{a_i == pi(x_i)} * y_i / P(a_i | x_i) ]
using only the randomized episodes. It needs no invented second outcome per episode. Rules are fixed on training rows
(the threshold baseline included) before being evaluated on held-out rows; CIs resample whole session/scene groups.
"""
import numpy as np


def _propensity(row):
    e = float(row['propensity'])
    if not 0 < e < 1:
        raise ValueError('Randomized assignment propensity required')
    return e if row['assigned'] else 1 - e


def ipw_value(rows, decisions):
    """Success estimate and help rate (share of episodes where the rule asks for help) for one rule."""
    if not rows:
        return None
    y = np.array([r['outcome'] == 'success' for r in rows], dtype=float)
    a = np.array([r['assigned'] for r in rows], dtype=int)
    w = np.array([1 / _propensity(r) for r in rows])
    d = np.asarray(decisions, dtype=int)
    match = (a == d).astype(float)
    # Self-normalized (Hajek) estimator: lower variance for small pilots; the plain estimate is reported too.
    plain = float(np.mean(match * w * y))
    normalized = float((match * w * y).sum() / max((match * w).sum(), 1e-12))
    return {'success_ipw': plain, 'success_ipw_normalized': normalized, 'help_rate': float(d.mean()), 'matched_episodes': int(match.sum())}


def threshold_rule(train, feature='tracking_error'):
    """Simple execution-error baseline: help when the feature exceeds a threshold chosen on TRAIN rows only."""
    values = [r['features'].get(feature) for r in train]
    finite = sorted(v for v in values if v is not None and np.isfinite(v))
    if not finite:
        return None
    best = (None, -1.)
    for threshold in np.unique(np.quantile(finite, np.linspace(0, 1, 11))):
        decisions = [int(r['features'].get(feature) is not None and r['features'][feature] > threshold) for r in train]
        value = ipw_value(train, decisions)['success_ipw_normalized']
        if value > best[1]:
            best = (float(threshold), value)
    return {'feature': feature, 'threshold': best[0], 'chosen_on': 'train rows only'}


def decisions_for(rule, rows, artifacts):
    from .research import predict
    if rule == 'never':
        return [0] * len(rows)
    if rule == 'always':
        return [1] * len(rows)
    if rule.startswith('threshold:'):
        spec = artifacts['_threshold']
        if spec is None:
            return [0] * len(rows)
        return [int(r['features'].get(spec['feature']) is not None and r['features'][spec['feature']] > spec['threshold']) for r in rows]
    if rule.startswith('risk:'):
        # Help when failure without help is predicted likely (p_without < 0.5), regardless of the predicted gain.
        return [int(p['p_without'] < .5) for p in predict(artifacts[rule[5:]], rows)]
    if rule.startswith('gain:'):
        return [int(p['gain'] > 0) for p in predict(artifacts[rule[5:]], rows)]
    raise ValueError('Unknown rule')


def evaluate(artifacts, train, held_out, seed=1000, bootstrap=500):
    """artifacts: {variant_name: fitted research artifact}. Returns per-rule IPW values with group-bootstrap CIs."""
    from .datasets import connected_groups
    held_out = [r for r in held_out if r['outcome'] in {'success', 'failure'}]
    if not held_out:
        return {'status': 'no_held_out_randomized_episodes'}
    artifacts = dict(artifacts)
    artifacts['_threshold'] = threshold_rule([r for r in train if r['outcome'] in {'success', 'failure'}])
    rules = ['never', 'always', 'threshold:tracking_error']
    for name in sorted(a for a in artifacts if not a.startswith('_')):
        rules += ['risk:' + name, 'gain:' + name]
    decisions = {rule: decisions_for(rule, held_out, artifacts) for rule in rules}
    point = {rule: ipw_value(held_out, decisions[rule]) for rule in rules}
    groups = connected_groups([dict(r, id=r['episode_id']) for r in held_out])
    index = {r['episode_id']: i for i, r in enumerate(held_out)}
    members = [[index[e['id']] for e in g] for g in groups]
    ci = None
    if len(members) > 1:
        rng = np.random.default_rng(seed)
        samples = {rule: [] for rule in rules}
        for _ in range(bootstrap):
            chosen = [i for g in rng.integers(0, len(members), len(members)) for i in members[g]]
            rows = [held_out[i] for i in chosen]
            for rule in rules:
                samples[rule].append(ipw_value(rows, [decisions[rule][i] for i in chosen])['success_ipw_normalized'])
        ci = {rule: np.percentile(v, [2.5, 97.5]).tolist() for rule, v in samples.items()}
    return {'status': 'completed', 'estimator': 'IPW (plain and self-normalized) on held-out randomized episodes',
            'random_rule_reference': 'a 50 % random rule equals the observed held-out success rate in expectation',
            'observed_success_rate': float(np.mean([r['outcome'] == 'success' for r in held_out])),
            'threshold_baseline': artifacts['_threshold'], 'rules': point, 'group_bootstrap_ci95_normalized': ci,
            'independent_groups': len(members), 'held_out_episodes': len(held_out),
            'limitation': 'Estimated success had the fixed rule been followed; not a causal effect for any single episode. '
                          'With few groups the CI is wide or unavailable; help_rate is the human-time cost proxy.'}
