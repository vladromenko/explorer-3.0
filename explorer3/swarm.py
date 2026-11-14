"""The fleet as one team: one text command, the robots share the work.

  «осмотрите комнату»  both mobile robots explore (frontiers split between them), the cameras
                       inventory objects on the way, everything goes to the shared map/memory;
  «найдите <X>»        both explore looking for X; the first sighting ends the search for both;
  «подъедь к <X>»      the robot that can get there (ROSMASTER if both) drives to it;
  «принеси <X>»        locate (memory, else team search) -> ROSMASTER drives to X, picks it up,
                       drives to the hand-over spot, puts it on the pad, backs off -> SO-101 takes
                       it from the pad and holds it out to the person.

Every team task is one episode (kind team_explore / team_find / fetch) with per-stage times, so
the article's numbers come from the runs themselves. Tank's session is opened by the swarm itself
(owner 'swarm', kept alive by its own heartbeat) — no «Включить управление» click needed.
"""
import math
import threading
import time

from .episodes import Episode
from .goto import to_prompt
from . import world_memory

NAMES = {'tank': 'Tank', 'rosmaster': 'ROSMASTER', 'so101': 'SO-101'}


def parse(text):
    """(kind, prompt) from a free Russian/English command."""
    t = text.lower().strip()
    if any(w in t for w in ('осмотр', 'исслед', 'обследу', 'explore', 'карту')):
        return 'explore', None
    if any(w in t for w in ('принес', 'привез', 'подай', 'дай ', 'bring', 'fetch')) or t.startswith('дай'):
        return 'fetch', to_prompt(t)
    if any(w in t for w in ('найд', 'ищи', 'где ', 'find', 'search')) or t.startswith('где'):
        return 'find', to_prompt(t)
    return 'goto', to_prompt(t)


class TankLink:
    """Gives the swarm Tank's drive loop: reuses an open session or opens one as owner 'swarm'."""

    OWNER = 'swarm'

    def __init__(self, control, session):
        self.control, self.session = control, session
        self.engaged = False
        self.beat = None

    def _alive(self):
        return self.session.thread is not None and self.session.thread.is_alive()

    def ensure(self, timeout=25.):
        c = self.control
        if c.robot == 'tank' and self._alive() and not c.expired():
            return                                      # the operator already opened it: share it
        if c.owner is not None and c.owner != self.OWNER:
            raise RuntimeError('Tank/SO-101 сейчас в ручном управлении')
        if c.robot != 'tank':
            c.select('tank')
        self.session.start(self.OWNER, 'tank_dock', 'swarm', record=False)
        self.engaged = True
        self.beat = threading.Thread(target=self._heartbeat, daemon=True, name='swarm-tank-heartbeat')
        self.beat.start()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self._alive():
                raise RuntimeError(self.session.error or c.last_stop_reason or 'Tank не включился')
            if self.session.phase not in ('preparing', 'idle'):
                return
            time.sleep(.2)
        raise RuntimeError('Tank не включился за отведённое время')

    def _heartbeat(self):
        while self.engaged:
            self.control.heartbeat(self.OWNER)
            time.sleep(.3)

    def start(self, goal):
        self.ensure()
        self.control.goal = goal

    def goal(self):
        return self.control.goal

    def cancel(self, reason):
        self.control.cancel_goal(reason)

    def release(self):
        self.engaged = False
        if self.control.owner == self.OWNER:
            self.control.stop('Рой: задание Tank завершено')


class RosLink:
    def __init__(self, runner):
        self.runner = runner

    def start(self, goal):
        end = time.monotonic() + 8
        while self.runner.running() and time.monotonic() < end:     # previous task is closing its session
            time.sleep(.1)
        self.runner.start(None, goal)

    def goal(self):
        return self.runner.goal if self.runner.running() else None

    def cancel(self, reason):
        self.runner.cancel(reason)

    def release(self):
        pass


class Swarm:
    def __init__(self, tank=None, ros=None, arm_factory=None, so101_factory=None, explore_factory=None,
                 goto_factory=None, drive_factory=None):
        self.links = {}
        if ros is not None:
            self.links['rosmaster'] = ros
        if tank is not None:
            self.links['tank'] = tank
        self.arm_factory = arm_factory
        self.so101_factory = so101_factory
        self.explore_factory = explore_factory
        self.goto_factory = goto_factory
        self.drive_factory = drive_factory
        self.thread = None
        self.stop_event = threading.Event()
        self.state = {'task': None, 'stage': None, 'message': 'Рой ждёт команду', 'log': [], 'active': False}
        self.current = {}                 # robot -> goal object of this task
        self.only = None
        self.skill = None                 # running arm / SO-101 skill (for STOP)
        self.handoff = None

    # ------------------------------------------------------------------ public API
    def command(self, text, robots=None):
        """robots: subset of mobile robots that take part (solo vs team runs for the article)."""
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('Рой уже выполняет задание: ' + (self.state['task'] or ''))
        kind, prompt = parse(text)
        if kind != 'explore' and not prompt:
            raise ValueError('Не понял, что искать')
        self.only = set(robots) if robots else None
        self.stop_event.clear()
        self.state = {'task': text, 'kind': kind, 'prompt': prompt, 'stage': 'start', 'message': 'Начинаю',
                      'log': [], 'active': True, 'robots': {}}
        self.thread = threading.Thread(target=self._run, args=(kind, prompt, text), daemon=True, name='swarm')
        self.thread.start()
        return kind, prompt

    def stop(self, reason='STOP роя'):
        self.stop_event.set()
        for robot, link in self.links.items():
            try:
                link.cancel(reason)
            except Exception:
                pass
        if self.skill is not None:
            self.skill.stop()

    def taken(self):
        if self.handoff is not None:
            self.handoff.taken.set()

    def status(self):
        robots = {}
        for robot, goal in list(self.current.items()):
            try:
                robots[robot] = goal.status()['message']
            except Exception:
                pass
        return {**self.state, 'robots': robots}

    # ------------------------------------------------------------------ helpers
    def say(self, stage, message):
        self.state['stage'], self.state['message'] = stage, message
        self.state['log'].append((round(time.time(), 1), stage, message))
        del self.state['log'][:-40]

    def _check(self):
        if self.stop_event.is_set():
            raise InterruptedError('Остановлено оператором')

    def _start(self, robot, goal):
        self.links[robot].start(goal)
        self.current[robot] = goal

    def _wait(self, goals, timeout, any_done=False):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._check()
            done = [g for g in goals if not g.active]
            if (any_done and done) or len(done) == len(goals):
                return
            time.sleep(.2)
        for g in goals:
            g.cancel('Время задания вышло')

    def mobile(self):
        return [r for r in ('rosmaster', 'tank') if r in self.links and (not self.only or r in self.only)]

    def _make_explore(self, robot, **kw):
        if self.explore_factory is not None:
            return self.explore_factory(robot, **kw)
        from .explore import Explore
        return Explore(robot, **kw)

    def _make_goto(self, robot, text):
        if self.goto_factory is not None:
            return self.goto_factory(robot, text)
        from .goto import GoTo
        return GoTo(robot, text)

    def _make_drive(self, robot, pose, text):
        if self.drive_factory is not None:
            return self.drive_factory(robot, pose, text)
        from .drive import DriveTo
        return DriveTo(robot, pose, text=text)

    def _team(self, **kw):
        """Start an explore task on every available mobile robot; returns {robot: goal}."""
        goals = {}
        for robot in self.mobile():
            try:
                goal = self._make_explore(robot, **kw)
                self._start(robot, goal)
                goals[robot] = goal
            except Exception as exc:
                self.say('team', f'{NAMES[robot]} не участвует: {exc}')
        if not goals:
            raise RuntimeError('Ни один мобильный робот не доступен')
        return goals

    # ------------------------------------------------------------------ tasks
    def _run(self, kind, prompt, text):
        episode = Episode('fleet', {'explore': 'team_explore', 'find': 'team_find', 'fetch': 'fetch', 'goto': 'team_goto'}[kind], text)
        stages = []
        outcome, message = 'error', ''
        try:
            fn = {'explore': self._explore, 'find': self._find, 'fetch': self._fetch, 'goto': self._goto}[kind]
            outcome, message = fn(prompt, text, stages)
        except InterruptedError as exc:
            outcome, message = 'cancelled', str(exc)
        except Exception as exc:
            outcome, message = 'error', f'{type(exc).__name__}: {exc}'
        finally:
            for robot, link in self.links.items():
                try:
                    goal = self.current.get(robot)
                    if goal is not None and goal.active:
                        goal.cancel('Задание роя завершено')
                    link.release()
                except Exception:
                    pass
            self.current = {}
            self.skill = self.handoff = None
            self.say(outcome, message)
            self.state['active'] = False
            episode.extra.update(stages=stages, robots=self.mobile(), condition='team' if len(self.mobile()) > 1 else 'solo')
            episode.finish({'done': 'arrived', 'explored': 'explored'}.get(outcome, outcome), message)

    def _stage(self, stages, name, message):
        self._check()
        stages.append({'stage': name, 'started': round(time.time(), 2)})
        self.say(name, message)

    def _end_stage(self, stages, ok, note=''):
        if stages:
            stages[-1].update(seconds=round(time.time() - stages[-1]['started'], 1), ok=ok, note=note)

    def _explore(self, prompt, text, stages):
        self._stage(stages, 'explore', 'Роботы вместе осматривают комнату и записывают, что видят')
        goals = self._team(scout=True, kind='explore', text='осмотр комнаты (рой)')
        self._wait(list(goals.values()), 900)
        found = sorted({i['prompt'] for i in world_memory.known_objects(max_age_s=1800)})
        self._end_stage(stages, True, ', '.join(found))
        from .fleet_map import FLEET
        contrib = FLEET.status().get('contrib', {})
        return 'explored', ('Комната осмотрена. Найдено: ' + (', '.join(found) or 'ничего из списка') +
                            ' · вклад в карту: ' + ', '.join(f'{NAMES.get(r, r)} {n}' for r, n in contrib.items()))

    def _known(self, prompt, max_age_s=3600):
        return world_memory.target_for(None, prompt, max_age_s)

    def _search(self, prompt, stages):
        """Team search until somebody has placed the object on the map. Returns the sighting."""
        started = time.time()
        self._stage(stages, 'search', f'Ищем «{prompt}» вдвоём')

        def found():
            item = world_memory.target_for(None, prompt)
            if item is not None and item['at'] >= started:
                return f'{NAMES.get(item["robot"], item["robot"])} нашёл «{prompt}»'
            return None

        goals = self._team(scout=True, look_for=prompt, stop_when=found, kind='explore', text=f'поиск {prompt} (рой)')
        self._wait(list(goals.values()), 900)
        item = world_memory.target_for(None, prompt)
        ok = item is not None and item['at'] >= started
        self._end_stage(stages, ok, item['robot'] if ok else '')
        return item if ok else None

    def _find(self, prompt, text, stages):
        item = self._search(prompt, stages)
        if item is None:
            return 'not_found', f'«{prompt}» не найден ни одним роботом'
        x, y = item['room_xy']
        return 'done', f'{NAMES.get(item["robot"], item["robot"])} нашёл «{prompt}» в точке ({x:.1f}; {y:.1f}) м'

    def _goto(self, prompt, text, stages):
        allowed = self.mobile()
        robot = 'rosmaster' if 'rosmaster' in allowed else allowed[0]
        item = self._known(prompt)
        if item is not None and len(allowed) > 1 and item.get('room_xy'):
            # the robot closer to the remembered place goes
            def dist(r):
                p = world_memory.POSES[r]
                return math.dist((p.x, p.y), item['room_xy']) if p.known() else 99.
            robot = min(allowed, key=dist)
        self._stage(stages, 'goto', f'{NAMES[robot]} едет к «{prompt}»')
        goal = self._make_goto(robot, text)
        self._start(robot, goal)
        self._wait([goal], 300)
        self._end_stage(stages, goal.state == 'arrived', goal.message)
        return ('done' if goal.state == 'arrived' else goal.state), goal.message

    def _fetch(self, prompt, text, stages):
        from . import so101_handoff
        missing = so101_handoff.missing()
        if missing:
            raise RuntimeError('Сначала один раз покажите места передачи: ' + '; '.join(missing))
        cfg = so101_handoff.load()
        # 1. where is it
        item = self._known(prompt, max_age_s=24 * 3600)
        if item is None or not item.get('room_xy'):
            item = self._search(prompt, stages)
            if item is None:
                return 'not_found', f'«{prompt}» не найден: принести нечего'
        # 2. ROSMASTER drives to it
        self._stage(stages, 'approach', f'ROSMASTER едет к «{prompt}»')
        if 'tank' in self.links:
            self.links['tank'].release()               # Tank's part (search) is done
        if 'rosmaster' not in self.links:
            raise RuntimeError('Принести может только ROSMASTER: он не подключён')
        goal = self._make_goto('rosmaster', 'к ' + prompt)
        self._start('rosmaster', goal)
        self._wait([goal], 300)
        self._end_stage(stages, goal.state == 'arrived', goal.message)
        if goal.state != 'arrived':
            return 'failed', 'ROSMASTER не доехал: ' + goal.message
        # 3. pick it up
        self._stage(stages, 'pick', f'ROSMASTER берёт «{prompt}»')
        arm = self.arm_factory(self.say) if self.arm_factory else None
        if arm is None:
            from .rosmaster_arm import ArmSkills
            arm = ArmSkills(log=lambda m: self.say('pick', m))
        self.skill = arm
        try:
            picked = arm.pick(prompt)
            self._end_stage(stages, picked['ok'], picked.get('reason', ''))
            if not picked['ok']:
                return 'failed', 'ROSMASTER не взял: ' + picked['reason']
            arm.release()
            # 4. to the hand-over spot
            self._stage(stages, 'deliver', 'ROSMASTER везёт к SO-101')
            drive = self._make_drive('rosmaster', cfg['rosmaster_spot'], 'к месту передачи')
            self._start('rosmaster', drive)
            self._wait([drive], 240)
            self._end_stage(stages, drive.state == 'arrived', drive.message)
            if drive.state != 'arrived':
                return 'failed', 'ROSMASTER не доехал до места передачи: ' + drive.message
            # 5. put it on the pad, back off
            self._stage(stages, 'place', 'ROSMASTER кладёт на площадку передачи')
            arm.place(cfg['rosmaster_place'], picked['grip'])
            arm.drive(-.15, 0.)
            self._end_stage(stages, True)
        finally:
            arm.close()
            self.skill = None
        # 6. SO-101 hands it over
        self._stage(stages, 'handoff', 'SO-101 берёт с площадки и протягивает вам')
        handoff = self.so101_factory(prompt, lambda m: self.say('handoff', m)) if self.so101_factory else \
            so101_handoff.Handoff(prompt, log=lambda m: self.say('handoff', m))
        self.handoff = self.skill = handoff
        result = handoff.run()
        self._end_stage(stages, result['ok'], result.get('reason', ''))
        if not result['ok']:
            return 'failed', result['reason']
        return 'done', f'«{prompt}» доставлено: ROSMASTER привёз, SO-101 передал'


SWARM = None


def install(control, session):
    """Create the app-wide swarm with real robots (called once by explorer3.app)."""
    global SWARM
    from .rosmaster_goto import runner
    SWARM = Swarm(tank=TankLink(control, session), ros=RosLink(runner))
    return SWARM
