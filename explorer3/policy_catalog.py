"""Explicit installed-policy recipes; readiness never means learned robot skill."""
from .paths import DATA
from .profiles import profile

SMOL_REV='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
VLM_REV='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
POLICIES={'act':'ACT · базовая политика','smolvla':'SmolVLA · зрение + язык + действия',
          'diffusion':'Diffusion Policy · последовательности действий'}


def smolvla_paths():
    root=DATA/'cache/huggingface/hub'
    base=root/'models--lerobot--smolvla_base/snapshots'/SMOL_REV
    vlm=root/'models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots'/VLM_REV
    for path in (base,vlm):
        if not (path/'config.json').is_file() or not (path/'model.safetensors').is_file():
            raise ValueError('Нет закреплённых локальных весов SmolVLA; сначала подготовьте модели')
    return base,vlm


def validate_recipe(policy,name):
    if policy not in POLICIES:raise ValueError('Unknown policy')
    cfg=profile(name)
    if policy!='act' and cfg.get('domain')!='sim' and name!='so101_pick_place':
        raise ValueError('Новая политика сначала проверяется в симуляции; аппаратный профиль пока закрыт')
    if policy=='smolvla':smolvla_paths()
