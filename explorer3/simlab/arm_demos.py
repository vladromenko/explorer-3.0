"""M6.3: scripted-teacher demonstrations for the arm in the arm_v1 contract (absolute targets).

Runs the existing privileged geometric teacher (explorer3.sim.expert.ScriptedSO101) in the
CPU MuJoCo environment and records (actor observation, normalized delta action) pairs in
exactly the layout of explorer3.simlab.arm_warp. Used to initialize PPO by behaviour
cloning; only successful episodes are kept.
"""
import numpy as np

from .arm_warp import MAX_STEP


def collect(episodes=200, seed=0, history=3, log=print, noise=0.3):
    from explorer3.sim.environment import SO101PickPlace
    from explorer3.sim.expert import ScriptedSO101
    env = SO101PickPlace()
    lo, hi = env.ranges[:, 0], env.ranges[:, 1]
    span = hi - lo
    norm = lambda q: 2 * (q - lo) / span - 1
    box = env.data.body('target_box').xpos.copy()
    rng = np.random.default_rng(seed)
    X, Y, kept = [], [], 0
    for ep in range(episodes):
        env.reset(seed=int(seed * 100000 + ep))
        expert = ScriptedSO101(env)
        command = env.data.ctrl[:6].copy()
        prev = norm(command)
        held = env.data.body('task_cube').xpos.copy()
        age = 0.0
        frames, actions = [], []
        for t in range(int(env.deadline_s * env.fps)):
            cube = env.data.body('task_cube').xpos.copy()
            visible = rng.random() > 0.05
            if visible:
                held = cube + rng.normal(0, 0.005, 3)
                age = 0.0
            else:
                age = min(1.0, age + env.dt)
            free_v = int(env.model.joint('cube_free').dofadr[0])
            obs = np.concatenate([norm(env.data.qpos[:6]), env.data.qvel[:6] / 3, held / 0.3,
                                  [float(visible), age], box / 0.3, [t / (env.deadline_s * env.fps)]]).astype(np.float32)
            expert.action()                              # advances the teacher's phase and solves its IK target
            # Label = the phase's IK target (depends on the cube/box and time, not on where the
            # arm currently is). The demo is executed through the SAME rate limiter the clone and
            # the robot use, with small execution noise (DART) so the clone also sees deviations.
            phase_target = np.clip(env.to_radians(np.asarray(expert.targets[expert.phase])), lo, hi)
            label = norm(phase_target)
            target = phase_target + rng.normal(0, noise, 6) * MAX_STEP * 3
            command = command + np.clip(target - command, -MAX_STEP, MAX_STEP)
            frames.append(obs)
            actions.append(label.astype(np.float32))
            prev = label
            _, _, term, trunc, info = env.step(env.from_radians(command))
            if term or trunc:
                break
        if info.get('outcome') == 'success':
            kept += 1
            f = np.asarray(frames)
            stacked = np.concatenate([np.concatenate([np.repeat(f[:1], max(0, history - 1 - i), 0), f[max(0, i - history + 1):i + 1]])
                                      .reshape(1, -1) for i in range(len(f))])
            X.append(stacked)
            Y.append(np.asarray(actions))
        if ep % 20 == 0:
            log(f'demo {ep}: kept {kept}')
    env.close()
    return np.concatenate(X), np.concatenate(Y), kept
