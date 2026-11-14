"""Scene measurements and scenario files (M3)."""
import hashlib
from functools import lru_cache

import yaml

from explorer3.paths import ROOT


@lru_cache(maxsize=None)
def load_measurements():
    return yaml.safe_load((ROOT / 'configs/scene_measurements.yaml').read_text())


@lru_cache(maxsize=None)
def load_scenario(name):
    return yaml.safe_load((ROOT / f'configs/scenarios/{name}.yaml').read_text())


def fingerprint(name):
    """Hash of the scenario and the measurements it depends on; stored in policy artifacts."""
    text = (ROOT / f'configs/scenarios/{name}.yaml').read_text() + (ROOT / 'configs/scene_measurements.yaml').read_text()
    return hashlib.sha256(text.encode()).hexdigest()


def assumed_fields():
    """List measurement sections still marked as assumed (shown in reports)."""
    return sorted(k for k, v in load_measurements().items() if isinstance(v, dict) and v.get('status') != 'measured')
