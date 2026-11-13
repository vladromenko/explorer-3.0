"""Atomic publication of PPO + official VecNormalize as one checked bundle."""
import hashlib
import json
import uuid
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(model,work):
    work=Path(work);base=work/'checkpoints';base.mkdir(exist_ok=True)
    name=f'step-{model.num_timesteps}-{uuid.uuid4().hex[:8]}'
    pending=base/('.'+name);pending.mkdir()
    model.save(pending/'policy.zip')
    model.get_env().save(pending/'normalization.pkl')
    manifest={'version':1,'steps':model.num_timesteps,'normalization':'SB3 VecNormalize',
              'files':{name:digest(pending/name) for name in ('policy.zip','normalization.pkl')}}
    (pending/'manifest.json').write_text(json.dumps(manifest,indent=2))
    final=base/name;pending.rename(final)
    pointer=work/'latest-bundle.json.tmp'
    pointer.write_text(json.dumps({'checkpoint':str(final/'policy.zip'),'steps':model.num_timesteps}))
    pointer.replace(work/'latest-bundle.json')
    return final/'policy.zip'


def verify(checkpoint):
    checkpoint=Path(checkpoint).resolve();manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    if checkpoint.name!='policy.zip' or set(manifest['files'])!={'policy.zip','normalization.pkl'}:
        raise ValueError('Invalid PPO normalization bundle')
    for name,expected in manifest['files'].items():
        if digest(checkpoint.parent/name)!=expected:raise ValueError('PPO/normalization checkpoint mismatch')
    return checkpoint.parent/'normalization.pkl'


def latest(work):
    checkpoint=Path(json.loads((Path(work)/'latest-bundle.json').read_text())['checkpoint'])
    if not checkpoint.resolve().is_relative_to((Path(work)/'checkpoints').resolve()):
        raise ValueError('Checkpoint pointer outside job')
    verify(checkpoint)
    return checkpoint
