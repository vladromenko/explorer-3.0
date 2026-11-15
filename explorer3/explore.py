"""«Исследовать комнату»: the robot drives to where the map ends (frontiers) until the whole
reachable room is mapped. Same interface as explorer3.goto.GoTo, so Tank's session loop
(control.goal) and ROSMASTER's runner drive it unchanged: command(rgb) -> [v, w].

Swarm behaviour:
  * robots that explore at the same time split the frontiers (a frontier another robot is
    heading for is skipped while there are others);
  * a robot that has not yet found itself on the shared map explores its own map, which is
    stitched into the shared one as soon as it recognises common walls (fleet_map);
  * with scout=True the camera inventories objects on the way (explorer3.scout);
  * stop_when() lets the team end the task early (e.g. the other robot found the object).
"""
import math
import time

from . import navigation
from .episodes import Episode
from .fleet_map import FLEET

NAMES = {'tank': 'Tank', 'rosmaster': 'ROSMASTER'}
ACTIVE = {}                 # robot -> running Explore (for frontier sharing)


def nav_now(robot, last_cmd, max_age=3.):
    """(frame, grid, pose) with the pose extrapolated by the last command since it was measured."""
    got = FLEET.nav(robot, max_age)
    if got is None:
        return None
    frame, grid, pose, at = got
    dt = min(1., max(0., time.time() - at))
    v, w = last_cmd
    x, y, t = pose
    return frame, grid, (x + v * dt * math.cos(t + w * dt / 2), y + v * dt * math.sin(t + w * dt / 2), navigation.wrap(t + w * dt))


def pose_now(robot, last_cmd, max_age=3.):
    got = nav_now(robot, last_cmd, max_age)
    return None if got is None or got[0] != 'main' else got[2]


class Explore:
    def __init__(self, robot, *, v_max=.15, w_max=.7, time_limit_s=600, record=True, scout=False,
                 look_for=None, stop_when=None, kind='explore', text='исследовать комнату', visual_search=False):
        self.robot = robot
        self.text = text
        self.prompt = 'explore'
        self.v_max, self.w_max, self.time_limit = v_max, w_max, time_limit_s
        self.active = True
        self.state, self.message = 'start', 'Начинаю: жду своё место на карте'
        self.started = time.monotonic()
        self.last_cmd = (0., 0.)
        self.target = self.path = None
        self.frame = None
        self.planned_at = 0.
        self.banned = []
        self.look_until = None
        self.look = None                 # where the camera should look on arrival
        self.visited = []                # reached targets (never retried, unlike banned ones)
        self.progress = (None, time.monotonic())
        self.lost_since = None
        self.targets_done = 0
        self.retried = False
        self.distance = self.bearing = None
        self.stop_when = stop_when
        self.visual = bool(look_for) or visual_search
        self.scout = None
        if scout:
            from .scout import Scout
            self.scout = Scout(robot, priority=look_for)
        self.episode = Episode(robot, kind, text) if record else None
        if record:
            ACTIVE[robot] = self

    def set_camera(self, cam):
        if self.scout is not None:
            self.scout.set_camera(cam)

    def command(self, rgb):
        if not self.active:
            return [0., 0.]
        if self.scout is not None:
            self.scout.feed(rgb)
        if self.stop_when is not None:
            why = self.stop_when()
            if why:
                self.finish('team_done', why)
                return [0., 0.]
        cmd = self._command()
        self.last_cmd = (cmd[0], cmd[1])
        if self.episode is None:
            return cmd
        self.episode.step(pose_now(self.robot, (0., 0.)), cmd, self.state, rgb,
                          target=None if self.target is None else [round(v, 2) for v in self.target], frame=self.frame)
        return cmd

    coordinate = True     # False: control condition A3 — no frontier split with the other robots

    def _peer_targets(self):
        if not self.coordinate:
            return []
        out = []
        for robot, other in list(ACTIVE.items()):
            if robot != self.robot and other.active and other.frame == 'main':
                at = pose_now(robot, other.last_cmd)
                out.append((None if at is None else at[:2], other.target))
        return out

    def _command(self):
        now = time.monotonic()
        if now - self.started > self.time_limit:
            self.finish('timeout', f'Время вышло: обследовано {self.targets_done} участков')
            return [0., 0.]
        got = nav_now(self.robot, self.last_cmd)
        if got is None:
            # No scan yet (or the robot is lost in both maps): turn slowly so the lidar sees more.
            self.lost_since = self.lost_since or now
            if now - self.lost_since > 40:
                self.finish('lost', 'Не могу определить своё положение')
                return [0., 0.]
            self.state, self.message = 'localizing', 'Определяю своё положение…'
            return [0., .35 if (now - self.lost_since) % 3 < 1.5 else 0.]
        self.lost_since = None
        frame, grid, pose = got
        if frame != self.frame:                          # own map stitched into the shared one, etc.
            self.frame, self.target, self.path = frame, None, None
            self.banned, self.visited = [], []
        where = 'общей карте' if frame == 'main' else 'своей карте (общие стены пока не видны)'
        if self.target is None or now - self.planned_at > 2.:
            self.planned_at = now
            fronts = navigation.frontiers(grid, self.robot, pose, banned=self.banned + self.visited, visual=self.visual and frame == 'main')
            if self.target is not None and not any(math.hypot(f[0] - self.target[0], f[1] - self.target[1]) < .4 for f in fronts):
                self.targets_done += 1                     # the place got mapped meanwhile (maybe by the other robot)
                self.target = None
            if self.target is None:
                if not fronts and self.banned and not self.retried:
                    self.banned, self.retried = [], True     # places given up earlier: one more try
                    return [0., 0.]
                if not fronts:
                    self.finish('explored', f'Обследовано: {self.targets_done} участков, на карте {FLEET.grid.scans} сканов')
                    return [0., 0.]
                peers = self._peer_targets() if frame == 'main' else []
                chosen = navigation.choose(fronts, peers)
                self.target = chosen[:2]
                self.look = chosen[4:6] if len(chosen) > 4 else None
                self.progress = (pose, now)
            path = navigation.plan(grid, self.robot, pose[:2], self.target, allow_unknown=False)
            if path is None:
                self.banned.append(self.target)
                self.target = None
                return [0., 0.]
            self.path = path
            if frame == 'main':
                FLEET.track.setdefault(self.robot, {})['path'] = list(path)
        ref, since = self.progress
        if ref is not None and math.dist(ref[:2], pose[:2]) > .15:
            self.progress = (pose, now)
        elif now - since > 10:
            # could not get there: «temporarily unreachable», NOT «seen» — the object may still be
            # there; the banned list is retried once before the search ends
            self.banned.append(self.target)
            self.target = None
            self.message = 'Не получается проехать: выбираю другое место'
            return [0., 0.]
        v, w, remaining = navigation.follow(pose, self.path, self.v_max, self.w_max)
        if remaining < .25:
            if self.visual and self.look is not None and math.dist(self.look, pose[:2]) > .25:
                # the unseen floor is beside the stop (a narrow gap): turn the camera to it first
                err = navigation.wrap(math.atan2(self.look[1] - pose[1], self.look[0] - pose[0]) - pose[2])
                if abs(err) > .3:
                    self.state, self.message = 'look', f'Поворачиваю камеру к непросмотренному месту: {math.degrees(err):+.0f}°'
                    return [0., max(-self.w_max, min(self.w_max, 1.5 * err))]
                # hold still long enough for one recognition pass (SAM 3: ~2.5 s for the list)
                self.look_until = self.look_until or now + (2.8 if self.scout is not None else .6)
                if now < self.look_until:
                    self.state, self.message = 'look', 'Смотрю в непросмотренное место'
                    return [0., 0.]
            self.look_until = None
            self.targets_done += 1
            # Arrived: if this boundary is still there it cannot be seen from the reachable side
            # (a pocket behind furniture) — do not come back to it.
            self.visited.append(self.target)
            if self.visual and frame == 'main':
                grid.mark_seen(self.target, .25)         # arrived: only the floor under the robot (the camera looks ahead)
                if self.look is not None:
                    grid.mark_seen(self.look, .15)       # looked at it: a gap seen from outside is done
            self.target = None
            return [0., 0.]
        v, w, why = navigation.guard(FLEET.scan(self.robot), self.robot, v, w)
        self.distance = remaining
        self.state = 'blocked' if why else 'explore'
        found = ''
        if self.scout is not None and self.scout.found:
            found = ' · нашёл: ' + ', '.join(sorted({self.scout.names.get(p, p) for p, _ in self.scout.found}))
        self.message = (f'Объезжаю: {why}' if why else
                        f'Еду к неизведанному месту по {where}: {remaining:.1f} м · участков: {self.targets_done}') + found
        return [v, w]

    def finish(self, state, message):
        self.state, self.message, self.active = state, message, False
        if ACTIVE.get(self.robot) is self:
            ACTIVE.pop(self.robot, None)
        if self.scout is not None:
            self.scout.stop()
        FLEET.track.setdefault(self.robot, {})['path'] = None
        if self.episode is not None:
            if self.scout is not None:
                self.episode.extra['found'] = sorted({p for p, _ in self.scout.found})
            self.episode.extra['frame'] = self.frame
            self.episode.finish(state, message)

    def cancel(self, reason='Отменено'):
        if self.active:
            self.finish('cancelled', reason)

    def status(self):
        return {'state': self.state, 'message': self.message, 'prompt': self.prompt, 'active': self.active,
                'distance': self.distance, 'bearing': None, 'box': None, 'score': None, 'detect_ms': None}
