"""Videos of policy rollouts in full MuJoCo: corner/overview camera + the robot's own camera
inset + captions. H.264 mp4 for the browser; one file per episode plus index.json."""
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

OUTCOME_RU = {'success': 'УСПЕХ', 'collision': 'СТОЛКНОВЕНИЕ', 'timeout': 'НЕ УСПЕЛ', 'failure': 'НЕУДАЧА'}
COLORS = {'success': (95, 212, 168), 'collision': (255, 123, 114), 'timeout': (242, 182, 90), 'failure': (255, 123, 114)}


_FONTS = {}


def _font(size, bold=False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        try:
            _FONTS[key] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)
        except OSError:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def _text(img, xy, text, size=15, color=(230, 237, 243), bold=False):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(xy, text, font=_font(size, bold), fill=color)
    return np.asarray(pil).copy()


def _caption(frame, lines, outcome=None):
    img = frame.copy()
    overlay = img.copy()
    cv2.rectangle(overlay, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (13, 17, 23), -1)
    img = cv2.addWeighted(overlay, .7, img, .3, 0)
    img = _text(img, (12, img.shape[0] - 26), '    '.join(l for l in lines if l), 15)
    if outcome:
        label = OUTCOME_RU.get(outcome, outcome)
        w = int(_font(26, True).getlength(label))
        cv2.rectangle(img, (14, 14), (40 + w, 56), (13, 17, 23), -1)
        img = _text(img, (27, 19), label, 26, COLORS.get(outcome, (230, 237, 243)), True)
    return img


def _inset(frame, small, label):
    h, w = small.shape[:2]
    x, y = frame.shape[1] - w - 12, 12
    frame[y:y + h, x:x + w] = small
    cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + h), (230, 237, 243), 1)
    return _text(frame, (x + 6, y + 4), label, 13)


def _write(path, frames, fps):
    import av
    with av.open(str(path), 'w') as c:
        s = c.add_stream('libx264', rate=fps)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = 'yuv420p'
        s.options = {'crf': '24', 'preset': 'veryfast', 'movflags': '+faststart'}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format='rgb24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def dock_videos(policy_dir, seeds, out_dir, curriculum=1.0, size=(400, 640)):
    from .artifact import Policy
    from .dock_mujoco import DockMujoco, render_option
    policy = Policy(policy_dir)
    env = DockMujoco(policy.robot, seed=0, curriculum=curriculum)
    big = mujoco.Renderer(env.model, height=size[0], width=size[1])
    small = mujoco.Renderer(env.model, height=size[0] // 3, width=size[1] // 3)
    opt = render_option()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for seed in seeds:
        env.g.manual_seed(int(seed))
        env.reset(torch.ones(1, dtype=torch.bool))
        obs, _ = env.observe()
        policy.reset()
        frames, outcome = [], 'timeout'
        for t in range(env.deadline_steps):
            a = torch.tensor(policy.act(obs[0].numpy()))[None]
            action = env.command(a, env.lidar_bins(noisy=True))
            collided = env.integrate()
            _, term, trunc, info = env.finish(action, collided, auto_reset=False)
            big.update_scene(env.data, camera='chase', scene_option=opt)
            small.update_scene(env.data, camera='front_cam', scene_option=opt)
            frame = _inset(big.render().copy(), small.render().copy(), 'камера робота')
            v = float(env.vel[0, 0])
            done = bool(info['success'][0]) or bool(info['collision'][0]) or bool(trunc[0])
            if bool(info['success'][0]):
                outcome = 'success'
            elif bool(info['collision'][0]):
                outcome = 'collision'
            frames.append(_caption(frame, [f't {t / 20:4.1f} c', f'до зоны {float(info["distance"][0]):.2f} м',
                                           f'скорость {v:+.2f} м/с', 'стоп по лидару' if bool(info['safety_stop'][0]) else ''],
                                   outcome if done else None))
            if done:
                frames += [frames[-1]] * 20          # hold the final frame for 1 s
                break
            obs, _ = env.observe()
        name = f'seed{seed}.mp4'
        _write(out_dir / name, frames, 20)
        cv2.imwrite(str(out_dir / f'seed{seed}.jpg'), cv2.cvtColor(frames[len(frames) * 2 // 3], cv2.COLOR_RGB2BGR))
        index.append({'file': name, 'poster': f'seed{seed}.jpg', 'seed': int(seed), 'outcome': outcome, 'seconds': round(len(frames) / 20, 1)})
    (out_dir / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return index


def arm_videos(policy_dir, seeds, out_dir, size=(400, 640)):
    from explorer3.sim.environment import SO101PickPlace
    from . import contract
    from .artifact import Policy
    from .arm_warp import MAX_STEP
    policy = Policy(policy_dir)
    env = SO101PickPlace()
    big = mujoco.Renderer(env.model, height=size[0], width=size[1])
    small = mujoco.Renderer(env.model, height=size[0] // 3, width=size[1] // 3)
    lo, hi = env.ranges[:, 0], env.ranges[:, 1]
    box = env.data.body('target_box').xpos.copy()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for seed in seeds:
        env.reset(seed=int(seed))
        policy.reset()
        command = env.data.ctrl[:6].copy()
        steps = int(env.deadline_s * env.fps)
        frames, outcome = [], 'failure'
        for t in range(steps):
            cube = env.data.body('task_cube').xpos.copy()
            obs = contract.build_arm(2 * (env.data.qpos[:6] - lo) / (hi - lo) - 1, env.data.qvel[:6], cube, 1, 0.0, box, t / steps)
            target = lo + (policy.act_target(obs) + 1) / 2 * (hi - lo)
            command = command + np.clip(target - command, -MAX_STEP, MAX_STEP)
            _, _, te, tr, info = env.step(env.from_radians(command))
            if t % 2 == 0 or te or tr:            # 15 fps video
                big.update_scene(env.data, camera='overview')
                small.update_scene(env.data, camera='wrist_cam')
                frame = _inset(big.render().copy(), small.render().copy(), 'камера на запястье')
                done = te or tr
                if done:
                    outcome = info.get('outcome') or 'failure'
                frames.append(_caption(frame, [f't {t / 30:4.1f} c', f'кубик выше стола на {max(0., cube[2] - .017) * 100:.1f} см',
                                               'в коробке' if info.get('inside_box') else ''], outcome if done else None))
            if te or tr:
                frames += [frames[-1]] * 15
                break
        name = f'seed{seed}.mp4'
        _write(out_dir / name, frames, 15)
        cv2.imwrite(str(out_dir / f'seed{seed}.jpg'), cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_RGB2BGR))
        index.append({'file': name, 'poster': f'seed{seed}.jpg', 'seed': int(seed), 'outcome': outcome, 'seconds': round(len(frames) / 15, 1)})
    (out_dir / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return index
