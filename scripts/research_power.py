"""Rough power check for the main real-robot comparison (no robot, no data, pure simulation of outcomes).

Question: with N randomized episodes of one target profile (help assigned with probability 0.5), how often does the
held-out IPW value of a targeted help rule exceed the value of a random 50 % rule with a 95 % CI above zero?
The targeted rule here uses an IDEAL known feature (no estimation error), so real power is LOWER than printed.

Usage: PYTHONPATH=. .venv/bin/python scripts/research_power.py [repeats]
"""
import sys
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def trial(rng, n, max_gain, bootstrap=300):
    x = rng.normal(size=n)
    p_without = sigmoid(-.8 * x)                    # the policy struggles more when x is large
    p_with = np.clip(p_without + max_gain * sigmoid(2 * x), 0, 1)   # help is worth more when x is large
    assigned = rng.integers(0, 2, n)
    y = rng.random(n) < np.where(assigned, p_with, p_without)
    rule = (x > 0).astype(int)                      # targeted rule, helps in 50 % of episodes like the random rule
    match = assigned == rule
    value = lambda idx: (y[idx] * match[idx]).sum() / max(match[idx].sum(), 1) - y[idx].mean()
    point = value(np.arange(n))
    samples = [value(rng.integers(0, n, n)) for _ in range(bootstrap)]
    return point, np.percentile(samples, 2.5) > 0


def main(repeats=200):
    rng = np.random.default_rng(0)
    print('max help gain | N randomized episodes | true advantage over random rule | power (95 % CI > 0)')
    for max_gain in (.2, .3, .4):
        for n in (40, 80, 160, 320, 640):
            results = [trial(rng, n, max_gain) for _ in range(repeats)]
            print(f'{max_gain:13.1f} | {n:21d} | {np.mean([r[0] for r in results]):31.3f} | {np.mean([r[1] for r in results]):.2f}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
