"""One simlab run end to end: train (or clone the teacher) -> evaluate in full MuJoCo -> videos.
Writes ~/.local/share/explorer3/simlab/active.json while running so the /learn panel shows it.

Usage: MUJOCO_GL=egl PYTHONPATH=. .venv-sim/bin/python -u scripts/simlab_pipeline.py <tank|rosmaster|so101> <check|full|clone|eval> [--policy DIR] [--name NAME]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.paths import DATA  # noqa: E402

PY = [sys.executable, '-u']
TITLES = {'check': 'проверочный прогон', 'full': 'полное обучение', 'clone': 'клонирование учителя', 'eval': 'оценка и видео'}


def step(title, args):
    print(json.dumps({'stage': title, 'at': time.time()}, ensure_ascii=False), flush=True)
    code = subprocess.call(PY + args, cwd=ROOT)
    if code:
        raise SystemExit(f'{title}: exit {code}')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('robot', choices=['tank', 'rosmaster', 'so101'])
    a.add_argument('kind', choices=list(TITLES))
    a.add_argument('--policy')
    a.add_argument('--name')
    a.add_argument('--log')
    args = a.parse_args()
    stamp = time.strftime('%m%d_%H%M')
    name = args.name or (f'so101_bc_{stamp}' if args.robot == 'so101' else f'{args.robot}_dock_{args.kind}_{stamp}')
    from explorer3.resources import ResourceLease
    lease = ResourceLease('compute')
    lease.__enter__()        # the real-training job queue waits while this holds the GPU (and vice versa)
    active = DATA / 'simlab' / 'active.json'
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(json.dumps({'pid': os.getpid(), 'robot': args.robot, 'kind': args.kind, 'name': name,
                                  'log': args.log, 'started': time.time()}, ensure_ascii=False))
    try:
        policy = args.policy or str(DATA / 'simlab' / 'policies' / name)
        if args.kind in ('check', 'full'):
            minutes = '10' if args.kind == 'check' else '240'
            step('обучение', ['scripts/simlab_train.py', args.robot, '--minutes', minutes, '--steps', '4000000000', '--name', name])
        elif args.kind == 'clone':
            step('клонирование учителя и оценка', ['scripts/simlab_arm_bc.py', '--name', name])
        if args.robot != 'so101':
            for curriculum in ('0.5', '1.0'):
                step(f'оценка в MuJoCo (сложность {curriculum})', ['scripts/simlab_eval.py', policy, '--curriculum', curriculum, '--episodes', '100'])
        step('видео', ['scripts/simlab_video.py', policy, '--episodes', '4'])
        print(json.dumps({'stage': 'готово', 'at': time.time()}, ensure_ascii=False), flush=True)
    finally:
        lease.__exit__(None, None, None)
        try:
            if json.loads(active.read_text()).get('pid') == os.getpid():
                active.unlink()
        except (OSError, ValueError):
            pass


if __name__ == '__main__':
    main()
