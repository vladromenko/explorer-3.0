import json
from .paths import ROOT
PROFILES=json.loads((ROOT/'configs/profiles.json').read_text())
TRAINING=json.loads((ROOT/'configs/training.json').read_text())

def profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError('Неизвестный профиль')
    return PROFILES[name]
