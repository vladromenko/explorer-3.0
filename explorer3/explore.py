"""«Исследовать комнату»: the robot drives to where the shared map ends (frontiers) until
the whole reachable room is mapped. Same interface as explorer3.goto.GoTo, so Tank's session
loop (control.goal) and ROSMASTER's runner drive it unchanged: command(rgb) -> [v, w]."""
import math
import time

from . import navigation
from .episodes import Episode
from .fleet_map import FLEET

NAMES = {'tank': 'Tank', 'rosmaster': 'ROSMASTER'}


def pose_now(robot, last_cmd, max_age=3.):
    """Map pose extrapolated with the last command since it was measured."""
    pose, at = FLEET.pose(robot, max_age)
    if pose is None:
        return None
    dt = min(1., max(0., time.time() - at))
    v, w = last_cmd
    x, y, t = pose
    return (x + v * dt * math.cos(t + w * dt / 2), y + v * dt * math.sin(t + w * dt / 2), navigation.wrap(t + w * dt))


class Explore:
    def __init__(self, robot, *, v_max=.15, w_max=.7, time_limit_s=600, record=True):
        self.robot = robot
        self.text = 'исследовать комнату'
        self.prompt = 'explore'
        self.v_max, self.w_max, self.time_limit = v_max, w_max, time_limit_s
        self.active = True
        self.state, self.message = 'start', 'Начинаю: жду своё место на общей карте'
        self.started = time.monotonic()
        self.last_cmd = (0., 0.)
        self.target = self.path = None
        self.planned_at = 0.
        self.banned = []
        self.progress = (None, time.monotonic())
        self.lost_since = None
        self.targets_done = 0
        self.distance = self.bearing = None
        self.episode = Episode(robot, 'explore', self.text) if record else None

    def set_camera(self, cam):
        pass

    def command(self, rgb):
        if not self.active:
            return [0., 0.]
        cmd = self._command()
        self.last_cmd = (cmd[0], cmd[1])
        if self.episode is None:
            return cmd
        pose = pose_now(self.robot, (0., 0.))
        self.episode.step(pose, cmd, self.state, rgb, target=None if self.target is None else [round(v, 2) for v in self.target])
        return cmd

    def _command(self):
        now = time.monotonic()
        if now - self.started > self.time_limit:
            self.finish('timeout', f'Время вышло: обследовано {self.targets_done} участков')
            return [0., 0.]
        pose = pose_now(self.robot, self.last_cmd)
        if pose is None:
            # Not on the map (yet / any more): turn slowly so the lidar sees more, give up after 40 s.
            self.lost_since = self.lost_since or now
            if now - self.lost_since > 40:
                self.finish('lost', 'Не могу найти себя на общей карте')
                return [0., 0.]
            self.state, self.message = 'localizing', 'Ищу себя на общей карте…'
            return [0., .35 if (now - self.lost_since) % 3 < 1.5 else 0.]
        self.lost_since = None
        grid = FLEET.grid
        # Choose / re-plan every 2 s.
        if self.target is None or now - self.planned_at > 2.:
            self.planned_at = now
            fronts = navigation.frontiers(grid, self.robot, pose, banned=self.banned)
            if self.target is not None and not any(math.hypot(f[0] - self.target[0], f[1] - self.target[1]) < .4 for f in fronts):
                self.targets_done += 1                     # the place got mapped meanwhile
                self.target = None
            if self.target is None:
                if not fronts:
                    self.finish('explored', f'Комната обследована: {self.targets_done} участков, {FLEET.grid.scans} сканов')
                    return [0., 0.]
                self.target = fronts[0][:2]
                self.progress = (pose, now)
            path = navigation.plan(grid, self.robot, pose[:2], self.target, allow_unknown=False)
            if path is None:
                self.banned.append(self.target)
                self.target = None
                return [0., 0.]
            self.path = path
            FLEET.track.setdefault(self.robot, {})['path'] = list(path)
        # Stuck: no progress for 10 s -> give that frontier up.
        ref, since = self.progress
        if ref is not None and math.dist(ref[:2], pose[:2]) > .15:
            self.progress = (pose, now)
        elif now - since > 10:
            self.banned.append(self.target)
            self.target = None
            self.message = 'Не получается проехать: выбираю другое место'
            return [0., 0.]
        v, w, remaining = navigation.follow(pose, self.path, self.v_max, self.w_max)
        if remaining < .25:
            self.targets_done += 1
            self.target = None
            return [0., 0.]
        v, w, why = navigation.guard(FLEET.scan(self.robot), self.robot, v, w)
        self.distance = remaining
        self.state = 'blocked' if why else 'explore'
        self.message = (f'Объезжаю: {why}' if why else
                        f'Еду к неизведанному месту: {remaining:.1f} м · обследовано участков: {self.targets_done}')
        return [v, w]

    def finish(self, state, message):
        self.state, self.message, self.active = state, message, False
        FLEET.track.setdefault(self.robot, {})['path'] = None
        if self.episode is not None:
            self.episode.finish(state, message)

    def cancel(self, reason='Отменено'):
        if self.active:
            self.finish('cancelled', reason)

    def status(self):
        return {'state': self.state, 'message': self.message, 'prompt': self.prompt, 'active': self.active,
                'distance': self.distance, 'bearing': None, 'box': None, 'score': None, 'detect_ms': None}
