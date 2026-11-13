import json
from .paths import ROOT
PROFILES=json.loads((ROOT/'configs/profiles.json').read_text())
SIM_PROFILES=json.loads((ROOT/'configs/sim-profiles.json').read_text())
TRAINING=json.loads((ROOT/'configs/training.json').read_text())

def profile(name: str) -> dict:
    if name in SIM_PROFILES:
        return SIM_PROFILES[name]
    if name not in PROFILES:
        raise ValueError('Неизвестный профиль')
    return PROFILES[name]
