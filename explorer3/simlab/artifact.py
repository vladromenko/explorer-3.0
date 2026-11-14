"""Policy artifact export and dependency-free runtime (M6.4, M10).

The artifact is a directory with policy.npz (actor weights + normalization) and
policy.json (contract version/hash, robot, history, scenario fingerprint, dynamics
parameters, metrics). The runtime uses only numpy, so the same file runs in .venv,
.venv-sim and on the Tank's Raspberry Pi without torch.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import contract


def export(model, meta, out_dir, *, scenario, scenario_fingerprint, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {'norm_mean': model.actor_norm.mean.cpu().numpy(), 'norm_var': model.actor_norm.var.cpu().numpy()}
    linear = [m for m in model.actor if hasattr(m, 'weight')]
    for i, layer in enumerate(linear):
        arrays[f'w{i}'] = layer.weight.detach().cpu().numpy()
        arrays[f'b{i}'] = layer.bias.detach().cpu().numpy()
    np.savez(out_dir / 'policy.npz', **arrays)
    digest = hashlib.sha256((out_dir / 'policy.npz').read_bytes()).hexdigest()
    info = {'format': 'explorer3.simlab.mlp_elu.v1', 'robot': meta['robot'], 'contract_version': contract.VERSION,
            'contract_hash': contract.contract_hash(meta['robot']), 'history': meta['history'],
            'layers': len(linear), 'scenario': scenario, 'scenario_fingerprint': scenario_fingerprint,
            'weights_sha256': digest, 'train': meta, **(extra or {})}
    (out_dir / 'policy.json').write_text(json.dumps(info, indent=1))
    return info


class Policy:
    """Deterministic actor: mean action, numpy only."""

    def __init__(self, path):
        path = Path(path)
        self.info = json.loads((path / 'policy.json').read_text())
        robot = self.info['robot']
        if self.info['contract_version'] != contract.VERSION or self.info['contract_hash'] != contract.contract_hash(robot):
            raise ValueError('Policy contract does not match this Explorer version')
        raw = (path / 'policy.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.info['weights_sha256']:
            raise ValueError('Policy weights checksum mismatch')
        z = np.load(path / 'policy.npz')
        self.mean, self.std = z['norm_mean'], np.sqrt(z['norm_var'] + 1e-8)
        self.layers = [(z[f'w{i}'], z[f'b{i}']) for i in range(self.info['layers'])]
        self.robot, self.k = robot, self.info['history']
        self.dim = contract.size(robot)
        self.reset()

    def reset(self):
        self.buf = None

    def act(self, observation):
        """observation: one contract vector. Returns normalized action in [-1, 1]."""
        o = np.asarray(observation, dtype=np.float32)
        if o.shape != (self.dim,):
            raise ValueError('Observation shape does not match the contract')
        if self.buf is None:
            self.buf = np.repeat(o[None], self.k, 0)
        else:
            self.buf = np.roll(self.buf, -1, 0)
            self.buf[-1] = o
        x = np.clip((self.buf.reshape(-1) - self.mean) / self.std, -10, 10)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return np.clip(x, -1, 1).astype(np.float32)
