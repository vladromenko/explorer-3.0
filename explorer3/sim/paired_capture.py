"""One physics rollout, two immutable datasets differing only in instruction text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from explorer3.paths import DATA, ROOT, offline_environment

KNOWN = 'rosmaster_floor_pick_knowledge_sim'
CONTROL = 'rosmaster_floor_pick_noknowledge_sim'


def control_copy(source):
    source = Path(source)
    meta = json.loads((source / 'manifest.json').read_text())
    if meta['profile'] != KNOWN or meta['domain'] != 'sim':
        raise ValueError('Expected knowledge simulation capture')
    target = source.with_name(source.name + '-control')
    if target.exists():
        existing = json.loads((target / 'manifest.json').read_text())
        if existing.get('paired_source') != source.name:
            raise ValueError('Different paired capture already exists')
        return target
    temporary = target.with_name(target.name + '.partial')
    if temporary.exists():
        shutil.rmtree(temporary)  # only an incomplete derivative; source is immutable
    temporary.mkdir()
    for path in source.iterdir():
        if path.suffix == '.png' or path.name == 'samples.jsonl':
            os.link(path, temporary / path.name)
    meta.update(profile=CONTROL, task='pick up the ' + meta['episode_parameters']['object']['name'],
                paired_source=source.name)
    (temporary / 'manifest.json').write_text(json.dumps(meta, indent=2))
    temporary.rename(target)
    return target


def collect_seed(seed, output, *, external_spool=False):
    output = Path(output)
    parent = DATA / 'checkpoints/sim-staging' if external_spool else DATA / 'staging/sim'
    if external_spool and not os.path.ismount(DATA / 'checkpoints'):
        raise ValueError('External checkpoint volume is not mounted; collection not started')
    from .factory import make, EXPERTS
    from .capture import SimCapture
    # A restart reuses a finished spool/import, including failures. It never
    # selects a second successful trajectory for only one of the conditions.
    if output.exists():
        record = json.loads(output.read_text())
        root = Path(record['spool'])
    else:
        env = make(KNOWN)
        try:
            env.reset(seed=seed)
            expert = EXPERTS[KNOWN](env)
            capture = SimCapture(env, seed, 'scripted_expert', spool_parent=parent)
            group = f'knowledge-pair-{seed}-{env.manifest["xml_sha256"]}'
            capture.meta.update(session=group, scene=group, paired_group=group)
            try:
                while True:
                    action = expert.action()
                    capture.add(env, action)
                    _, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            except Exception as exc:
                capture.finish('aborted', f'technical_error:{type(exc).__name__}:{exc}')
                raise
            root = capture.finish(info['outcome'], info['termination_reason'])
            record = {'seed': seed, 'outcome': info['outcome'], 'spool': str(root),
                      'scene': group, 'model': env.manifest['xml_sha256'],
                      'object': env.episode_parameters['object'], 'frames': capture.count,
                      'samples_sha256': hashlib.sha256((root / 'samples.jsonl').read_bytes()).hexdigest()}
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(output)
        finally:
            env.close()
    counterpart = control_copy(root)
    for path in (root, counterpart):
        subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'explorer3.sim.import_episode', str(path)],
                       check=True, env=offline_environment())
    record['imports'] = {name: json.loads((path / 'imported.json').read_text())
                         for name, path in ((KNOWN, root), (CONTROL, counterpart))}
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(output)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external-spool', action='store_true')
    args = parser.parse_args()
    print(json.dumps(collect_seed(args.seed, args.output, external_spool=args.external_spool)), flush=True)
