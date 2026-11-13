"""IPW value of fixed help rules matches the analytic truth on randomized data."""
import numpy as np
from explorer3.research_value import ipw_value, threshold_rule


def rows(n=20000, seed=1):
    rng = np.random.default_rng(seed); out = []; truth = {'always': [], 'never': []}
    sig = lambda z: 1/(1+np.exp(-z))
    for _ in range(n):
        te = float(rng.gamma(2, .2)); a = int(rng.integers(0, 2))
        p1, p0 = sig(-.5+te), sig(-.2-2.5*te)
        out.append({'assigned': a, 'propensity': .5, 'outcome': 'success' if rng.random() < (p1 if a else p0) else 'failure',
                    'features': {'tracking_error': te}})
        truth['always'].append(p1); truth['never'].append(p0)
    return out, {k: float(np.mean(v)) for k, v in truth.items()}


def test_ipw_matches_truth_and_threshold_learned_on_train():
    data, truth = rows()
    assert abs(ipw_value(data, [1]*len(data))['success_ipw_normalized'] - truth['always']) < .015
    assert abs(ipw_value(data, [0]*len(data))['success_ipw_normalized'] - truth['never']) < .015
    rule = threshold_rule(data[:10000])
    assert rule['chosen_on'] == 'train rows only' and rule['threshold'] < .2
