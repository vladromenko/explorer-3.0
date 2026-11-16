"""«Рой без роботов»: the real swarm code (shared map, frontier split, camera-coverage search,
«принеси» with its stages and episode) in a virtual room, from the same /swarm page.

What is simulated: the room (explorer3.sim2d), the robots' motion and wheel odometry (ROSMASTER
3 % noise, Tank none), the lidars (ray casting, same mounts), the cameras (an object within 2.5 m
and ±35° in line of sight is «seen», as SAM 3 would report it), ROSMASTER's arm (takes the object
when it stands within 0.55 m of it) and SO-101 (hands it over after a pause).
What is real: everything else — FleetMap, Explore, DriveTo, navigation, Swarm, world memory,
episodes. Real robot data are not touched: the map file, the object memory, the episodes and the
hand-over config are swapped for simulation copies while the mode is on and restored after.
"""
import math
import shutil
import threading
import time

import numpy as np

from . import sim2d
from .paths import DATA

SIM = DATA / 'simlab' / 'swarm'
OBJECTS = ['cookie', 'sock', 'cube', 'bottle', 'cup', 'ball']
DT = .1
WORLD = None                       # the running SimWorld, or None (real robots)


class SimWorld:
    def __init__(self, seed=None, robots=('rosmaster', 'tank'), active_robots=None):
        from . import fleet_map as fm
        self.seed = int(time.time()) % 100000 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.active_robots = tuple(robots if active_robots is None else active_robots)
        if not self.active_robots or not set(self.active_robots).issubset(robots):
            raise ValueError('Simulation participants must be a nonempty subset of the layout robots')
        self.room = sim2d.make_room(self.rng)
        starts = []
        for _ in robots:
            starts.append(sim2d.free_spot(self.room, self.rng, avoid=starts))
        self.true = {r: (float(p[0]), float(p[1]), float(self.rng.uniform(-math.pi, math.pi))) for r, p in zip(robots, starts)}
        self.odom = dict(self.true)
        self.cmd = {r: (0., 0.) for r in robots}
        self.objects = {}
        for name in OBJECTS:
            self.objects[name] = sim2d.free_spot(self.room, self.rng, clearance=.3, avoid=starts)
        # SO-101's table with the hand-over pad: next to a wall, reachable
        self.pad = sim2d.free_spot(self.room, self.rng, clearance=.45, avoid=starts)
        self.carried = None
        self.delivered = []
        self.collisions = 0
        self.last_seen = {}
        # object sizes (m). MODELLED ASSUMPTION of the 2D swarm simulation, to be measured on the real
        # robots: a single-view estimate of the size is off by ~±40 %, and the grasp needs the width
        # within ±1.5 cm — so wide/flat objects need knowledge from the SO-101 station.
        self.sizes = {'cookie': .075, 'sock': .085, 'cube': .04, 'bottle': .07, 'cup': .085, 'ball': .065}
        # Independent streams: helper measurements must not consume navigation/sensor noise.
        self.grasp_rng = np.random.default_rng(self.seed + 104729)
        self.measure_rng = np.random.default_rng(self.seed + 130363)
        self.relocate_after_seen = set()        # «a person moved it» once, 5 s after the first sighting
        self.vanish_after_seen = set()          # «a person took it away»: help cannot succeed
        self.vanished = set()
        self.relocations = []
        self.lock = threading.Lock()
        self.links = {}
        self.stop_event = threading.Event()
        self.thread = None
        self.mounts = {r: fm.MOUNTS[r] for r in robots}
        # map frame = the first scanning robot's start (FleetMap convention): ROSMASTER scans first
        self.first = robots[0]

    # ------------------------------------------------------------------ frames
    def to_map(self, x, y, th=0.):
        from .fleet_map import inverse, compose
        return compose(inverse(self.true0[self.first]), (x, y, th))

    def handover_config(self):
        """Hand-over spot 0.35 m in front of the pad, facing it; SO-101 poses are placeholders."""
        start = self.true0[self.first]
        a = math.atan2(start[1] - self.pad[1], start[0] - self.pad[0])
        spot = (self.pad[0] + .35 * math.cos(a), self.pad[1] + .35 * math.sin(a), a + math.pi)
        for k in range(24):                              # keep the spot off the walls
            if sim2d.wall_distance(self.room, np.array(spot[:2])) > .3:
                break
            a += math.pi / 12
            spot = (self.pad[0] + .35 * math.cos(a), self.pad[1] + .35 * math.sin(a), a + math.pi)
        from .so101_handoff import POSES
        return {'rosmaster_spot': list(self.to_map(*spot)), 'rosmaster_place': [90, 120, 0, 0, 90, 30],
                'so101': {p: [0.] * 6 for p in POSES}, 'simulation': True}

    def floor_goal(self, prompt):
        """A separate reachable floor patch for the 2D development mission only."""
        from .mission_goal import first_floor_goal
        if prompt not in self.objects:
            raise ValueError('Предмет отсутствует в виртуальной комнате')
        origin = self.objects[prompt]
        for angle in np.linspace(0., 2 * math.pi, 16, endpoint=False):
            center = origin + .7 * np.array([math.cos(angle), math.sin(angle)])
            heading = math.atan2(center[1] - self.true['rosmaster'][1], center[0] - self.true['rosmaster'][0])
            approach = center - .22 * np.array([math.cos(heading), math.sin(heading)])
            if sim2d.wall_distance(self.room, center) > .35 and sim2d.wall_distance(self.room, approach) > .3:
                return {**first_floor_goal('sim'), 'center_xy_m': list(map(float, self.to_map(*center)[:2])),
                        'approach_pose': list(map(float, self.to_map(*approach, heading))),
                        'place_pose': [90., 120., 0., 0., 90., 30.],
                        'calibration': 'nominal 2D release model only'}
        raise ValueError('Рядом с предметом нет свободной зоны 40×40 см')

    # ------------------------------------------------------------------ physics loop
    def start(self):
        self.true0 = dict(self.true)
        self.thread = threading.Thread(target=self._run, daemon=True, name='swarm-sim')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def _run(self):
        from .fleet_map import FLEET
        from . import world_memory
        tick = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            robots = self.active_robots
            if tick % 4 == 0:                            # lidars at 2.5 Hz, as the real localizer
                for r in robots:
                    try:
                        FLEET.update(r, sim2d.lidar_packet(self.room, self.mounts[r], self.true[r], self.rng),
                                     self.odom[r] if r.startswith('rosmaster') else None)
                    except Exception:
                        pass
            if tick % 5 == 0:                            # cameras at 2 Hz
                for r in robots:
                    if not world_memory.POSES[r].known():
                        continue
                    for name, xy in list(self.objects.items()):
                        if name == self.carried:
                            continue
                        seen = sim2d.visible(self.room, self.true[r], xy)
                        if seen is not None:
                            if time.time() - self.last_seen.get((r, name), 0) > 3:
                                if name in self.vanish_after_seen and name not in self.vanished:
                                    self.vanished.add(name)
                                    threading.Timer(5., lambda n=name: self.objects.pop(n, None)).start()
                                if name in self.relocate_after_seen and not any(n == name for n, _ in self.relocations):
                                    threading.Timer(5., self.relocate, args=(name,)).start()
                                    self.relocations.append((name, None))
                                self.last_seen[(r, name)] = time.time()
                                world_memory.record(r, 'симуляция', name, seen[0] + self.rng.normal(0, .03),
                                                    seen[1] + self.rng.normal(0, .02), score=.9)
            for r in robots:
                link = self.links.get(r)
                goal = link.current if link is not None else None
                v, w = (0., 0.)
                if goal is not None and goal.active:
                    try:
                        v, w = goal.command(None)[:2]
                    except Exception as exc:
                        goal.cancel(f'Ошибка: {exc}')
                self.move(r, v, w, DT)
            tick += 1
            self.stop_event.wait(max(0., DT - (time.monotonic() - started)))

    def relocate(self, name):
        """A person moved the object: a free place no robot sees right now."""
        if name == self.carried or name not in self.objects:
            return None
        for _ in range(200):
            p = sim2d.free_spot(self.room, self.rng, clearance=.3)
            if all(sim2d.visible(self.room, pose, p) is None and np.linalg.norm(p - np.array(pose[:2])) > 1.5
                   for pose in self.true.values()):
                self.objects[name] = p
                self.relocations = [(n, q) for n, q in self.relocations if n != name] + [(name, p.tolist())]
                return p
        return None

    def move(self, r, v, w, dt):
        from .fleet_map import BODY, wrap
        with self.lock:
            x, y, th = self.true[r]
            nx, ny, nth = x + v * dt * math.cos(th), y + v * dt * math.sin(th), wrap(th + w * dt)
            if sim2d.wall_distance(self.room, np.array([nx, ny])) < BODY[r][3] - .01:
                self.collisions += 1
                nx, ny = x, y
            self.true[r] = (nx, ny, nth)
            ox, oy, oth = self.odom[r]
            ov, ow = v * (1 + self.rng.normal(0, .03)), w * (1 + self.rng.normal(0, .02))
            self.odom[r] = (ox + ov * dt * math.cos(oth), oy + ov * dt * math.sin(oth), wrap(oth + ow * dt))
            if self.carried and r == 'rosmaster':
                self.objects[self.carried] = np.array([nx + .2 * math.cos(nth), ny + .2 * math.sin(nth)])


class SimLink:
    """Same interface as swarm.TankLink / RosLink: the world's loop drives the goal."""
    def __init__(self, world, robot):
        self.world, self.robot = world, robot
        self.current = None
        world.links[robot] = self

    def start(self, goal):
        if self.current is not None and self.current.active:
            self.current.cancel('Новое задание')
        self.current = goal

    def goal(self):
        return self.current

    def cancel(self, reason):
        if self.current is not None and self.current.active:
            self.current.cancel(reason)

    def release(self):
        pass


class SimArm:
    """ROSMASTER's arm in the virtual room."""
    def __init__(self, world, log):
        self.world, self.log = world, log
        self.stopped = False

    def pick(self, prompt, attempts=3, knowledge=None):
        """As the real ArmSkills.pick: the arm camera must see it; the base corrects into the reach
        window (0.16-0.26 m ahead), up to three attempts; then the gripper closes on it."""
        from .rosmaster_arm import REACH_X, base_correction
        if getattr(self, 'execution_selection', 'geometric_baseline') != 'geometric_baseline':
            return {'ok': False, 'reason': '2D-комната не исполняет обученную политику; нужна её проверка в MuJoCo'}
        w = self.world
        if prompt not in w.objects:
            return {'ok': False, 'reason': f'«{prompt}» нет в комнате'}
        for attempt in range(attempts):
            time.sleep(.8)                                   # recognition pass
            pose = w.true['rosmaster']
            if sim2d.visible(w.room, pose, w.objects[prompt], reach=1.2, fov_deg=35) is None:
                return {'ok': False, 'reason': 'предмет не виден камерой руки'}
            x, y, th = pose
            dx, dy = w.objects[prompt] - np.array([x, y])
            fx, fy = math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy
            reach = math.hypot(fx, fy)
            if REACH_X[0] <= reach <= REACH_X[1] and abs(fy) <= .05:
                time.sleep(1.)
                size = w.sizes.get(prompt, .04)
                known = ((knowledge or {}).get('params') or {}).get('width_m')
                guess = known if known else .04 * (1 + w.grasp_rng.uniform(-.4, .4))      # single view, no knowledge
                if abs(guess - size) > .015:
                    self.log(f'захват промахнулся: ширину «{prompt}» оценил в {guess * 100:.1f} см, на деле {size * 100:.1f} см')
                    return {'ok': False, 'reason': 'неверная оценка ширины предмета'}
                w.carried = prompt
                self.log(f'Взял «{prompt}» (был в {reach:.2f} м)')
                return {'ok': True, 'grip': 40}
            forward, turn = base_correction((fx, fy))
            self.log(f'захват: подъезжаю {forward:+.2f} м, поворот {math.degrees(turn):+.0f}°')
            self.turn(turn)
            self.drive(forward, 0.)
        return {'ok': False, 'reason': 'не удалось поставить предмет в зону захвата'}

    def turn(self, angle):
        for _ in range(int(abs(angle) / .4 / DT)):
            self.world.move('rosmaster', 0., math.copysign(.4, angle), DT)
            time.sleep(DT)

    def release(self):
        pass

    def place(self, pose, grip):
        time.sleep(1.)
        w = self.world
        if getattr(self, 'terminal_goal', None):
            x, y, theta = w.true['rosmaster']
            # Modelled 2D release point, not a teleport to the requested goal.
            w.objects[w.carried] = np.array([x + .22 * math.cos(theta), y + .22 * math.sin(theta)])
        else:
            w.objects[w.carried] = np.array(w.pad, float)
        w.carried = None

    def observe_delivery(self, prompt):
        w = self.world
        xy = w.objects.get(prompt)
        if xy is None:
            return None
        return {'domain': 'sim', 'object_class': prompt, 'frame': 'fleet_map',
                'xy_m': list(map(float, w.to_map(*xy)[:2])), 'position_sigma_m': 0.,
                'object_half_extent_m': w.sizes[prompt] / 2,
                'at': time.time(), 'released': w.carried != prompt, 'inside_height_band': True,
                'source': 'sim2d_object_state', 'limitation': '2D state, not MuJoCo or physical verification'}

    def drive(self, dx, w_):
        for _ in range(int(abs(dx) / .1 / DT)):
            self.world.move('rosmaster', math.copysign(.1, dx), 0., DT)
            time.sleep(DT)

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class SimHandoff:
    def __init__(self, world, prompt, log):
        self.world, self.prompt, self.log = world, prompt, log
        self.taken = threading.Event()
        self.stopped = False

    def run(self):
        for msg in ('Беру с площадки', 'Протягиваю вам'):
            self.log(msg)
            time.sleep(1.5)
        self.log('Жду «Забрал» (виртуальный человек забирает через 2 с)')
        if not self.taken.wait(2):
            self.taken.set()                     # the virtual person takes it
        self.world.objects.pop(self.prompt, None)
        self.world.delivered.append(self.prompt)
        return {'ok': True}

    def stop(self):
        self.stopped = True
        self.taken.set()


def approach(world, robot, text):
    """GoTo for the virtual room: drive to 0.3 m before the remembered object, facing it."""
    from .drive import DriveTo
    from .goto import to_prompt
    from .explore import pose_now
    from . import world_memory
    prompt = to_prompt(text)
    item = world_memory.target_for(robot, prompt, 24 * 3600)
    if item is None:
        raise RuntimeError(f'«{prompt}» ещё не видели')
    from . import navigation
    from .fleet_map import FLEET
    ox, oy = item['room_xy']
    pose = pose_now(robot, (0., 0.)) or (0., 0., 0.)
    # the free side of the object: 0.3 m from it, not inside furniture/wall inflation, nearest to us
    blocked, free, _ = navigation.layers(FLEET.grid, robot)
    best = None
    for k in range(16):
        a = math.atan2(oy - pose[1], ox - pose[0]) + (k + 1) // 2 * (1 if k % 2 else -1) * math.pi / 8
        sx, sy = ox - .3 * math.cos(a), oy - .3 * math.sin(a)
        c = navigation._cell((sx, sy))
        if 0 <= c[0] < blocked.shape[1] and 0 <= c[1] < blocked.shape[0] and free[c[1], c[0]] and not blocked[c[1], c[0]]:
            best = (sx, sy, a)
            break
    if best is None:
        a = math.atan2(oy - pose[1], ox - pose[0])
        best = (ox - .3 * math.cos(a), oy - .3 * math.sin(a), a)
    goal = DriveTo(robot, best, text=f'к «{prompt}»', tol=.15, heading_tol=.25)
    goal.prompt = prompt
    return goal


# ---------------------------------------------------------------------- enter / leave
_saved = {}


def enter(seed=None, robots=('rosmaster', 'tank')):
    """Switch the swarm page to the virtual room. Returns the simulated Swarm."""
    global WORLD
    robots = tuple(robots)
    if not robots or len(set(robots)) != len(robots) or not set(robots) <= {'rosmaster', 'tank'}:
        raise ValueError('Unknown or duplicate simulation participant')
    from . import swarm, fleet_map, world_memory, episodes, so101_handoff, explore
    if WORLD is not None:
        return swarm.SWARM
    real = swarm.SWARM
    if real is not None and real.state.get('active'):
        raise RuntimeError('Рой выполняет задание на настоящих роботах')
    was_running = fleet_map.LOCALIZER.running()
    fleet_map.LOCALIZER.stop()                    # wait: a reader mid-request must not feed the virtual map
    for t in fleet_map.LOCALIZER.threads:
        t.join(timeout=15)
    if fleet_map.LOCALIZER.running():
        fleet_map.LOCALIZER.done.clear()
        raise RuntimeError('Лидарные потоки не остановились, попробуйте ещё раз')
    if was_running:
        fleet_map.FLEET.save()
    SIM.mkdir(parents=True, exist_ok=True)
    real_map = fleet_map.MAP_DIR / 'map.npz'
    if real_map.exists():
        shutil.copy2(real_map, SIM / 'real-map.npz')
    _saved.update(localizer=was_running, swarm=real, memory=world_memory.MEMORY, episodes=episodes.ROOT, handoff=so101_handoff.CONFIG,
                  poses={r: p.as_dict() for r, p in world_memory.POSES.items()})
    world_memory.MEMORY = SIM / 'memory.json'
    world_memory.forget()
    from . import knowledge as _knowledge
    _saved['knowledge'] = _knowledge.STORE
    _knowledge.STORE = SIM / 'object_knowledge.json'
    _knowledge.STORE.unlink(missing_ok=True)
    episodes.ROOT = SIM / 'episodes'
    so101_handoff.CONFIG = SIM / 'handoff.json'
    with fleet_map.FLEET.lock:
        fleet_map.FLEET.grid = fleet_map.Grid()
        fleet_map.FLEET.track.clear()
        fleet_map.FLEET.last_key.clear()
        fleet_map.FLEET.subs.clear()
        fleet_map.FLEET.contrib.clear()
    for p in world_memory.POSES.values():
        p.__init__(p.robot)
    explore.ACTIVE.clear()
    # Generate the same room/starts/objects for paired conditions, but an absent helper must
    # supply neither camera sightings nor lidar scans (including while standing still).
    world = SimWorld(seed, active_robots=robots)
    world.start()
    so101_handoff.save(world.handover_config())

    def explore_factory(robot, **kw):
        kw['scout'] = False                       # the virtual cameras report sightings themselves
        return explore.Explore(robot, **kw)

    from .so101_station import SimSO101Station
    swarm.SWARM = swarm.Swarm(tank=SimLink(world, 'tank') if 'tank' in robots else None,
                              ros=SimLink(world, 'rosmaster') if 'rosmaster' in robots else None,
                              station_factory=lambda log: SimSO101Station(world, log),
                              arm_factory=lambda say: SimArm(world, lambda m: say('pick', m)),
                              so101_factory=lambda prompt, log: SimHandoff(world, prompt, log),
                              explore_factory=explore_factory,
                              goto_factory=lambda robot, text: approach(world, robot, text))
    WORLD = world
    return swarm.SWARM


def leave():
    """Back to the real robots: the real map, memory, episodes and hand-over config return."""
    global WORLD
    from . import swarm, fleet_map, world_memory, episodes, so101_handoff, explore
    if WORLD is None:
        return
    swarm.SWARM.stop('Выход из симуляции')
    t = swarm.SWARM.thread
    if t is not None:
        t.join(timeout=5)
    WORLD.stop()
    WORLD = None
    swarm.SWARM = _saved['swarm']
    world_memory.MEMORY = _saved['memory']
    from . import knowledge as _knowledge
    _knowledge.STORE = _saved.get('knowledge', _knowledge.STORE)
    episodes.ROOT = _saved['episodes']
    so101_handoff.CONFIG = _saved['handoff']
    explore.ACTIVE.clear()
    with fleet_map.FLEET.lock:
        fleet_map.FLEET.grid = fleet_map.Grid()
        fleet_map.FLEET.track.clear()
        fleet_map.FLEET.last_key.clear()
        fleet_map.FLEET.subs.clear()
        fleet_map.FLEET.contrib.clear()
    for p in world_memory.POSES.values():
        p.__init__(p.robot)
    backup = SIM / 'real-map.npz'
    if backup.exists():
        fleet_map.MAP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, fleet_map.MAP_DIR / 'map.npz')
        fleet_map.FLEET.load()
    if _saved.get('localizer'):
        fleet_map.LOCALIZER.start()


def truth_overlay():
    """Walls and objects of the virtual room in the map frame (for the page), or None."""
    w = WORLD
    if w is None:
        return None
    segs = [(w.to_map(*a)[:2], w.to_map(*b)[:2]) for a, b in w.room['segs']]
    objs = {n: w.to_map(*xy)[:2] for n, xy in list(w.objects.items())}
    return {'walls': segs, 'objects': objs, 'pad': w.to_map(*w.pad)[:2], 'carried': w.carried,
            'delivered': list(w.delivered), 'collisions': w.collisions, 'seed': w.seed}
