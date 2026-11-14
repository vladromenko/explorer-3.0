"""Drive to a pose on the shared map (e.g. the hand-over spot) along a planned path, then turn to
the taught heading. Same goal interface as GoTo/Explore: command(rgb) -> [v, w]."""
import math
import time

from . import navigation
from .episodes import Episode
from .explore import pose_now
from .fleet_map import FLEET


class DriveTo:
    def __init__(self, robot, pose, *, text='к месту', tol=.10, heading_tol=.08, v_max=.12, w_max=.6,
                 time_limit_s=180, record=False):
        self.robot, self.goal = robot, tuple(pose)
        self.text, self.prompt = text, 'drive'
        self.tol, self.heading_tol, self.v_max, self.w_max = tol, heading_tol, v_max, w_max
        self.time_limit = time_limit_s
        self.started = time.monotonic()
        self.active = True
        self.state, self.message = 'start', f'Еду {text}'
        self.path, self.planned_at = None, 0.
        self.last_cmd = (0., 0.)
        self.distance = None
        self.episode = Episode(robot, 'drive', text) if record else None

    def set_camera(self, cam):
        pass

    def command(self, rgb):
        if not self.active:
            return [0., 0.]
        cmd = self._command()
        self.last_cmd = (cmd[0], cmd[1])
        if self.episode is not None:
            self.episode.step(pose_now(self.robot, (0., 0.)), cmd, self.state, rgb)
        return cmd

    def _command(self):
        now = time.monotonic()
        if now - self.started > self.time_limit:
            self.finish('timeout', f'Не доехал {self.text} за отведённое время')
            return [0., 0.]
        pose = pose_now(self.robot, self.last_cmd)
        if pose is None:
            self.state, self.message = 'localizing', 'Жду своё место на общей карте…'
            return [0., 0.]
        dist = math.dist(pose[:2], self.goal[:2])
        self.distance = dist
        if dist < self.tol:
            err = navigation.wrap(self.goal[2] - pose[2])
            if abs(err) < self.heading_tol:
                self.finish('arrived', f'На месте {self.text}')
                return [0., 0.]
            self.state, self.message = 'align', f'Разворачиваюсь {self.text}: {math.degrees(err):+.0f}°'
            return [0., max(-.4, min(.4, 1.2 * err))]
        if self.path is None or now - self.planned_at > 2.:
            self.planned_at = now
            path = navigation.plan(FLEET.grid, self.robot, pose[:2], self.goal[:2], allow_unknown=True)
            if path is None:
                self.finish('no_path', f'Нет пути {self.text} по карте')
                return [0., 0.]
            self.path = path
            FLEET.track.setdefault(self.robot, {})['path'] = list(path)
        v, w, remaining = navigation.follow(pose, self.path, self.v_max, self.w_max, lookahead=min(.12, self.tol))
        if remaining < self.tol:
            v = 0.
        v, w, why = navigation.guard(FLEET.scan(self.robot), self.robot, v, w)
        self.state = 'blocked' if why else 'drive'
        self.message = f'Объезжаю: {why}' if why else f'Еду {self.text}: {remaining:.1f} м'
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
