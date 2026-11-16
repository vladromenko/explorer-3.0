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
import json
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

    @staticmethod
    def ensure():
        """Like TankLink: the swarm starts ROSMASTER's agent after a reboot and takes the serial
        port from the native bridge itself — no «Передать управление Explorer» click."""
        from . import rosmaster_control
        rosmaster_control.ensure_agent()
        if not rosmaster_control.request('/health', timeout=4).get('hardware_open'):
            rosmaster_control.request('/acquire', {}, 45)

    def start(self, goal):
        self.ensure()
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
                 goto_factory=None, drive_factory=None, station_factory=None):
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
        self.station_factory = station_factory
        self.thread = None
        self.stop_event = threading.Event()
        self.state = {'task': None, 'stage': None, 'message': 'Рой ждёт команду', 'log': [], 'active': False}
        self.current = {}                 # robot -> goal object of this task
        self.only = None
        self.skill = None                 # running arm / SO-101 skill (for STOP)
        self.handoff = None
        from .fleet_agents import FleetAgents

        def domain():
            from . import swarm_sim
            return 'sim' if swarm_sim.WORLD is not None else 'real'
        self.agents = FleetAgents(self, domain)
        if _on_sighting not in world_memory.LISTENERS:
            world_memory.LISTENERS.append(_on_sighting)

    # ------------------------------------------------------------------ public API
    def search_trial(self, prompt, condition, scene, time_limit_s=180):
        """Experiment A (one robot vs two): a «найдите» attempt under a fixed protocol.
        A1 = ROSMASTER alone; A2 = ROSMASTER + Tank with frontier split; A3 = both, no split (control)."""
        robots = {'A1': ['rosmaster'], 'A2': ['rosmaster', 'tank'], 'A3': ['rosmaster', 'tank']}[condition]
        self.trial = {'condition': condition, 'scene': scene, 'time_limit_s': float(time_limit_s), 'prompt': prompt,
                      'started': time.time(), 'coordinate': condition != 'A3'}
        from .goto import STEMS
        stems = {english: stem for stem, english in STEMS}
        return self.command('найдите ' + stems.get(prompt, prompt), robots)

    def command(self, text, robots=None, *, executor=None, task_context=None):
        """robots: subset of mobile robots that take part (solo vs team runs for the article)."""
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('Рой уже выполняет задание: ' + (self.state['task'] or ''))
        kind, prompt = parse(text)
        if kind != 'explore' and not prompt:
            raise ValueError('Не понял, что искать')
        if executor not in (None, 'geometric_baseline'):
            raise ValueError('Unknown grasp executor')
        self.pick_executor = executor
        self.task_context = dict(task_context or {})
        if kind == 'fetch' and executor is None and 'terminal_goal' not in self.task_context:
            from .mission_goal import first_floor_goal
            from . import swarm_sim
            self.task_context['terminal_goal'] = (swarm_sim.WORLD.floor_goal(prompt) if swarm_sim.WORLD is not None
                                                  else first_floor_goal(self.agents.domain()))
        self.only = set(robots) if robots else None
        self.agents.start(kind, prompt, self.mobile())      # plan of registered skills, or a clear refusal
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
        return {**self.state, 'robots': robots, 'agents': self.agents.status()}

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

    def _team(self, robots=None, **kw):
        """Start an explore task on every available mobile robot; returns {robot: goal}."""
        goals = {}
        for robot in robots or self.mobile():
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
            self.agents.finish(outcome, message)
            trial, self.trial = getattr(self, 'trial', None), None
            if trial is not None:
                from . import research_log
                found = world_memory.target_for(None, trial['prompt'])
                research_log.record_trial(trial, outcome, message, time.time() - trial['started'],
                                          found if found and found['at'] >= trial['started'] else None,
                                          episode.id, self.mobile())
            self.state['active'] = False
            episode.extra.update(terminal_goal=getattr(self, 'task_context', {}).get('terminal_goal'), stages=stages, robots=self.mobile(), condition='team' if len(self.mobile()) > 1 else 'solo')
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
        for r in goals:
            self.agents.step('explore', r, 'идёт')
        self._wait(list(goals.values()), 900)
        for r, g in goals.items():
            self.agents.step('explore', r, 'готово' if g.state == 'explored' else g.state)
        found = sorted({i['prompt'] for i in world_memory.known_objects(max_age_s=1800)})
        self._end_stage(stages, True, ', '.join(found))
        from .fleet_map import FLEET
        contrib = FLEET.status().get('contrib', {})
        return 'explored', ('Комната осмотрена. Найдено: ' + (', '.join(found) or 'ничего из списка') +
                            ' · вклад в карту: ' + ', '.join(f'{NAMES.get(r, r)} {n}' for r, n in contrib.items()))

    def _known(self, prompt, max_age_s=3600):
        return world_memory.target_for(None, prompt, max_age_s)

    def _search(self, prompt, stages, robots=None):
        """Team search until somebody has placed the object on the map. Returns the sighting."""
        started = time.time()
        try:
            from .fleet_map import FLEET
            FLEET.forget_seen()                  # the object may have moved: the cameras look again
        except Exception:
            pass
        self._stage(stages, 'search', f'Ищем «{prompt}» ' + ('вдвоём' if len(robots or self.mobile()) > 1 else f'({NAMES[(robots or self.mobile())[0]]})'))

        def found():
            # any fresh sighting counts — also from a robot that does not know its place on the map
            # yet (then the object has no room position, the camera still saw it)
            item = latest_sighting(prompt, started)
            if item is not None:
                return f'{NAMES.get(item["robot"], item["robot"])} нашёл «{prompt}»'
            return None

        trial = getattr(self, 'trial', None)
        limit = trial['time_limit_s'] if trial else 900
        goals = self._team(robots, scout=True, look_for=prompt, stop_when=found, kind='explore', text=f'поиск {prompt} (рой)',
                           time_limit_s=limit)
        if trial and not trial['coordinate']:
            for g in goals.values():
                g.coordinate = False
        for r in goals:
            self.agents.step('search', r, 'идёт')
        self._wait(list(goals.values()), limit + 5)
        item = world_memory.target_for(None, prompt)
        if item is None or item['at'] < started:
            item = latest_sighting(prompt, started)
        ok = item is not None and item['at'] >= started
        for r in goals:
            self.agents.step('search', r, ('нашёл' if ok and item['robot'] == r else 'готово') if ok else 'не нашёл')
        self._end_stage(stages, ok, item['robot'] if ok else '')
        return item if ok else None

    def _find(self, prompt, text, stages):
        item = self._search(prompt, stages)
        if item is None:
            return 'not_found', f'«{prompt}» не найден ни одним роботом'
        if not item.get('room_xy'):
            return 'done', (f'{NAMES.get(item["robot"], item["robot"])} видит «{prompt}» в {item["distance"]:.1f} м '
                            f'(место на общей карте пока неизвестно)')
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
        context = getattr(self, 'task_context', {})
        terminal_goal = context.get('terminal_goal')
        missing = so101_handoff.missing() if terminal_goal is None else []
        cfg = so101_handoff.load() if terminal_goal is None else {}
        if 'rosmaster' not in self.links:
            raise RuntimeError('Принести может только ROSMASTER: он не подключён')
        # 0. what does ROSMASTER not know? (knowledge gaps -> who can fill them)
        from . import knowledge
        domain = self.agents.domain()
        executor = getattr(self, 'pick_executor', None)
        context = getattr(self, 'task_context', {})
        gaps = knowledge.assess('rosmaster', prompt, domain, context=context, executor=executor)
        self.agents.gaps(gaps, self.mobile())
        stages.append({'stage': 'skill_readiness', 'executor': executor, 'readiness': gaps['readiness'],
                       'registered_skill': gaps['registered_skill'].get('skill_id')})
        if gaps['readiness']['physically_infeasible'] or 'skill' in gaps['gaps']:
            from .cognitive.broker import prepare_acquisition
            request = prepare_acquisition(self.agents.mission, text, 'rosmaster', prompt, gaps, domain=domain, context=context)
            self.state['acquisition'] = request
            self.agents.role('LearningAndExperimentCurator').note(request['state'], request['title'], request_id=request['id'])
            return 'awaiting_skill', request['title'] + '. ' + request['operator_next']
        if terminal_goal is not None:
            from .mission_goal import preparation_reason
            reason = preparation_reason(terminal_goal, domain)
            if reason:
                return 'awaiting_delivery_setup', reason + ' Подъём не будет отмечен как доставка.'
        advisor = getattr(self, 'advisor', 'off')       # 'off' | 'decide': the language model chooses the gaps (validated)
        if advisor == 'decide':
            from .cognitive.gap_advisor import advise
            self.say('learn', 'Языковая модель оценивает, каких знаний не хватает…')
            t0 = time.time()
            advice = advise('rosmaster', prompt, gaps, self.mobile() + ['so101'])
            stages.append({'stage': 'advise', 'started': round(t0, 2), 'seconds': round(time.time() - t0, 1),
                           'ok': 'error' not in advice, 'note': json.dumps({'rules': gaps['gaps'], 'model': advice}, ensure_ascii=False)[:600]})
            self.agents.role('MissionCoordinator').note('модель', f'правила: {gaps["gaps"]}; модель: {advice.get("gaps", advice.get("error"))}')
            if 'error' not in advice:
                gaps = {**gaps, 'gaps': list(dict.fromkeys(gaps['gaps'] + [g for g in advice['gaps'] if g in ('where', 'how')]))}
        grasp_knowledge = gaps['how']
        knowledge_case = None
        allowed = getattr(self, 'helpers', None)          # experiment conditions: which helpers may be called
        learning = None
        if executor == 'geometric_baseline' and 'how' in gaps['gaps'] and (allowed is None or 'so101' in allowed):
            # the SO-101 station studies the object on its table WHILE the mobile robots search
            self.say('learn', f'ROSMASTER не знает, как брать «{prompt}»: SO-101 измеряет; проверенный опыт захвата учитывается отдельно')
            station = self.station_factory(lambda m: self.say('learn', m)) if self.station_factory else None
            if station is None:
                from .so101_station import SO101Station
                station = SO101Station(log=lambda m: self.say('learn', m))
            learning = {'started': time.time()}
            def learn():
                try:
                    learning['result'] = station.practice(prompt)
                except Exception as exc:
                    learning['result'] = {'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}
            learning['thread'] = threading.Thread(target=learn, daemon=True, name='so101-station')
            learning['thread'].start()
        # 1. where is it
        item = self._known(prompt, max_age_s=knowledge.WHERE_MAX_AGE_S) if 'where' not in gaps['gaps'] else None
        if item is None or not item.get('room_xy'):
            item = self._search(prompt, stages)
            if item is None:
                return 'not_found', f'«{prompt}» не найден: принести нечего'
        else:
            self.agents.step('search', 'rosmaster', 'из памяти')
            self.agents.step('search', 'tank', 'не нужен')
        if learning is not None:
            learning['thread'].join(timeout=180)
            got = learning.get('result') or {'ok': False, 'reason': 'стенд SO-101 не ответил за 3 мин'}
            stages.append({'stage': 'learn', 'started': round(learning['started'], 2),
                           'seconds': round(time.time() - learning['started'], 1), 'ok': got['ok'], 'note': got.get('reason', '')})
            if got['ok']:
                grasp_knowledge = got['knowledge']
                knowledge_case = self.agents.knowledge_help('so101', 'rosmaster', prompt, grasp_knowledge)
            else:
                self.say('learn', 'Без знания о предмете: ' + got['reason'] + ' — беру по своей оценке')
        evidence = [e for e in [self.agents.sighting(item)] if e]
        if 'tank' in self.links:
            self.links['tank'].release()               # Tank's part (search) is done
        arm = None
        pending = []                                   # (case, helper outcome) waiting for the recipient's result
        picked = None
        try:
            for attempt in range(1 + 2):
                # 2. ROSMASTER drives to it (the arm hands the chassis back first)
                if arm is not None:
                    arm.release()
                self._stage(stages, 'approach', f'ROSMASTER едет к «{prompt}»' + (f' (попытка {attempt + 1})' if attempt else ''))
                self.agents.step('approach', 'rosmaster', 'идёт')
                goal = self._make_goto('rosmaster', 'к ' + prompt)
                self._start('rosmaster', goal)
                self._wait([goal], 300)
                arrived = goal.state == 'arrived'
                self._end_stage(stages, arrived, goal.message)
                self.agents.step('approach', 'rosmaster', 'готово' if arrived else 'не доехал', goal.message)
                failure = None if arrived else ('object_not_reached', 'approach', 'ROSMASTER не доехал: ' + goal.message)
                if arrived:
                    # 3. pick it up
                    self._stage(stages, 'pick', f'ROSMASTER берёт «{prompt}»')
                    self.agents.step('pick_floor', 'rosmaster', 'идёт')
                    if arm is None:
                        arm = self.arm_factory(self.say) if self.arm_factory else None
                        if arm is None:
                            from .rosmaster_arm import ArmSkills
                            arm = ArmSkills(log=lambda m: self.say('pick', m))
                    self.skill = arm
                    arm.execution_selection = ('geometric_baseline' if executor == 'geometric_baseline'
                                               else gaps['registered_skill'].get('job_id'))
                    picked = arm.pick(prompt, knowledge=grasp_knowledge)
                    if grasp_knowledge:
                        knowledge.record_use(grasp_knowledge['id'], 'rosmaster', picked['ok'], picked.get('reason', ''))
                    self._end_stage(stages, picked['ok'], picked.get('reason', ''))
                    self.agents.step('pick_floor', 'rosmaster', 'держит' if picked['ok'] else 'не взял', picked.get('reason', ''))
                    if not picked['ok']:
                        failure = ('pick_failed', 'pick_floor', 'ROSMASTER не взял: ' + picked['reason'])
                # recipient's continuation after each earlier help, measured by this attempt
                pending[:] = [e if len(e) > 2 else (e[0], e[1], 'success' if failure is None else 'failure',
                                                     failure[2] if failure else 'ROSMASTER держит предмет') for e in pending]
                if failure is None:
                    break
                reason, skill, message = failure
                decision = self.agents.assist('rosmaster', skill, prompt, reason, self.mobile(), evidence)
                if decision['option'] == 'human':
                    self._settle(pending, 'failure')
                    return 'failed', message + ' · нужен человек'
                before = time.time()
                if decision['option'] == 'helper_observe':
                    donor = decision['donor']
                    self._stage(stages, 'help', f'{NAMES[donor]} помогает: ищет «{prompt}» заново')
                    old = item.get('room_xy')
                    found = self._search(prompt, stages, robots=[donor])
                    helper = 'success' if found is not None and found['at'] >= before else 'failure'
                    if found is not None:
                        item = found
                        evidence = [e for e in [self.agents.sighting(found)] if e] or evidence
                        if old:
                            self.agents.conflict(prompt, old, found['room_xy'])
                    if donor == 'tank' and 'tank' in self.links:
                        self.links['tank'].release()
                    if decision['case']:
                        pending.append((decision['case'], helper))
                    if helper == 'failure':
                        self._settle(pending, 'failure')
                        return 'failed', message + f' · {NAMES[donor]} тоже не нашёл «{prompt}»'
                elif decision['case']:
                    pending.append((decision['case'], 'success'))      # retry: the recipient is its own helper
            else:
                self._settle(pending, 'failure')
                return 'failed', message
            if terminal_goal is not None:
                if knowledge_case:
                    pending.append((knowledge_case, 'success', 'success', 'ROSMASTER взял предмет'))
                outcome, note = self._deliver_floor(prompt, terminal_goal, arm, picked, stages)
                self._settle(pending, 'success' if outcome == 'done' else 'unknown')
                return outcome, note
            if missing:
                self.agents.step('deliver', 'rosmaster', 'не нужен: передача не настроена')
                if knowledge_case:
                    pending.append((knowledge_case, 'success', 'success', 'ROSMASTER взял по знанию SO-101'))
                self._settle(pending, 'success')
                return 'done', f'ROSMASTER взял «{prompt}» и держит его (передача SO-101 не настроена)'
            arm.release()
            # 4. to the hand-over spot
            self._stage(stages, 'deliver', 'ROSMASTER везёт к SO-101')
            self.agents.step('deliver', 'rosmaster', 'идёт')
            drive = self._make_drive('rosmaster', cfg['rosmaster_spot'], 'к месту передачи')
            self._start('rosmaster', drive)
            self._wait([drive], 240)
            self._end_stage(stages, drive.state == 'arrived', drive.message)
            self.agents.step('deliver', 'rosmaster', 'готово' if drive.state == 'arrived' else 'не доехал')
            if drive.state != 'arrived':
                self._settle(pending, 'failure')
                return 'failed', 'ROSMASTER не доехал до места передачи: ' + drive.message
            # 5. put it on the pad, back off
            self._stage(stages, 'place', 'ROSMASTER кладёт на площадку передачи')
            arm.place(cfg['rosmaster_place'], picked['grip'])
            arm.drive(-.15, 0.)
            self._end_stage(stages, True)
            self.agents.step('place', 'rosmaster', 'готово')
        finally:
            if arm is not None:
                arm.close()
            self.skill = None
        # 6. SO-101 hands it over
        self._stage(stages, 'handoff', 'SO-101 берёт с площадки и протягивает вам')
        self.agents.step('handover', 'so101', 'идёт')
        handoff = self.so101_factory(prompt, lambda m: self.say('handoff', m)) if self.so101_factory else \
            so101_handoff.Handoff(prompt, log=lambda m: self.say('handoff', m))
        self.handoff = self.skill = handoff
        result = handoff.run()
        self._end_stage(stages, result['ok'], result.get('reason', ''))
        taken = handoff.taken.is_set() if hasattr(handoff, 'taken') else False
        self.agents.step('handover', 'so101', ('забрали' if taken else 'протянул') if result['ok'] else 'не смог')
        # the person's «Забрал» is the measurement of the whole task; without it the outcome is unknown
        task_outcome = 'success' if result['ok'] and taken else 'failure' if not result['ok'] else 'unknown'
        if knowledge_case:
            pending.append((knowledge_case, 'success', 'success' if picked and picked['ok'] else 'failure',
                            'ROSMASTER взял по знанию SO-101' if picked and picked['ok'] else 'не взял и со знанием'))
        self._settle(pending, task_outcome)
        if not result['ok']:
            return 'failed', result['reason']
        return 'done', f'«{prompt}» доставлено: ROSMASTER привёз, SO-101 передал'

    def _deliver_floor(self, prompt, zone, arm, picked, stages):
        """Use the existing drive/place executors, with a separate terminal measurement."""
        if not callable(getattr(arm, 'observe_delivery', None)):
            return 'grasp_only', 'Захват завершён; независимая проверка выкладки недоступна. Доставка не подтверждена.'
        arm.release()
        self._stage(stages, 'deliver', 'ROSMASTER везёт в отмеченную зону на полу')
        drive = self._make_drive('rosmaster', zone['approach_pose'], 'к конечной зоне')
        self._start('rosmaster', drive)
        self._wait([drive], 240)
        self._end_stage(stages, drive.state == 'arrived', drive.message)
        if drive.state != 'arrived':
            return 'failed', 'ROSMASTER не доехал до конечной зоны: ' + drive.message
        self._stage(stages, 'place', 'ROSMASTER кладёт в отмеченную зону')
        arm.terminal_goal = zone
        started = time.time()
        arm.place(zone['place_pose'], picked['grip'])
        self._check()
        observed = arm.observe_delivery(prompt)
        from .mission_goal import delivery_verified
        verified = delivery_verified(zone, observed, domain=self.agents.domain(), object_class=prompt, since=started)
        stages[-1]['observation'] = observed
        self._end_stage(stages, verified)
        self.agents.step('deliver', 'rosmaster', 'доехал')
        self.agents.step('place', 'rosmaster', 'проверено' if verified else 'не подтверждено')
        if verified:
            return 'done', f'«{prompt}» оставлен в отмеченной зоне на полу; выкладка проверена'
        return 'delivery_unverified', 'Выкладка выполнена, но предмет в конечной зоне независимо не подтверждён'

    def _settle(self, pending, task):
        """OutcomeVerifier: helper / recipient / task outcomes of every help in this mission."""
        for entry in pending:
            case, helper = entry[0], entry[1]
            recipient = entry[2] if len(entry) > 2 else 'unknown'
            measurement = entry[3] if len(entry) > 3 else 'получатель не продолжил'
            self.agents.verify(case, helper, recipient, task, measurement)
        pending.clear()


SWARM = None


def latest_sighting(prompt, since):
    """Newest sighting of `prompt` recorded after `since` (with or without a room position)."""
    items = [i for i in world_memory._load() if i['prompt'] == prompt and i['at'] >= since]
    return max(items, key=lambda i: i['at']) if items else None


def _on_sighting(item):
    """Every sighting goes to the SceneObserver of the swarm that is running now (real or virtual)."""
    team = SWARM
    if team is not None and team.state.get('active'):
        team.agents.sighting(item)


def install(control, session):
    """Create the app-wide swarm with real robots (called once by explorer3.app)."""
    global SWARM
    from .rosmaster_goto import runner
    from .cognitive import store
    store.initialize()                       # research memory tables (idempotent)
    SWARM = Swarm(tank=TankLink(control, session), ros=RosLink(runner))
    return SWARM
