"""/learn — one training panel for real demonstrations (LeRobot ACT/SmolVLA/Diffusion) and
MuJoCo simlab policies: readiness of the data, presets with time estimates, two-click
start, live progress, loss curves, logs and follow-up actions in one place."""
import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from nicegui import ui

from . import catalog, datasets, training_drafts
from .paths import DATA, ROOT
from .policy_catalog import POLICIES, validate_recipe
from .profiles import PROFILES

CSS = r'''
body{background:#0d1117!important}
.lx{max-width:1320px;margin:0 auto;padding:22px 18px 60px;color:#e6edf3;font-family:Inter,"Segoe UI",system-ui,sans-serif;box-sizing:border-box;width:100%}
.lx *{box-sizing:border-box}
.lx a{color:#5fd4c0;text-decoration:none}
.lx-top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:6px}
.lx-title{font-size:30px;font-weight:750;letter-spacing:-.5px}
.lx-sub{color:#8b98a8;font-size:14px;margin-top:2px}
.lx-chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{padding:5px 11px;border-radius:999px;font-size:13px;background:#161d26;border:1px solid #263243;color:#c9d4e0;white-space:nowrap}
.chip.ok{color:#5fd4a8;border-color:#1f5a47;background:#0f2a22}
.chip.busy{color:#f2b65a;border-color:#5c4416;background:#2a2010}
.chip.bad{color:#ff7b72;border-color:#6b2a26;background:#2a1312}
.chip.info{color:#79b8ff;border-color:#1d3b5c;background:#0f1d2e}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;width:100%}
.grid2{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px;width:100%;align-items:start}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
.card{background:#141b24;border:1px solid #232f3e;border-radius:16px;padding:16px 18px;width:100%}
.card.click{cursor:pointer;transition:border-color .15s,transform .15s}
.card.click:hover{border-color:#3a4d63;transform:translateY(-1px)}
.card.sel{border-color:#2fb39f;box-shadow:0 0 0 1px #2fb39f inset;background:#122123}
.card h3{margin:0;font-size:16px;font-weight:650;line-height:1.3}
.muted{color:#8b98a8;font-size:13px;line-height:1.45}
.num{font-size:28px;font-weight:750;line-height:1.1}
.bar{height:8px;background:#222d3b;border-radius:6px;overflow:hidden;width:100%}
.bar>div{height:100%;background:linear-gradient(90deg,#24a08f,#63e0b8);border-radius:6px}
.bar.warn>div{background:linear-gradient(90deg,#c88a2a,#f2c46a)}
.step{display:flex;gap:12px;align-items:flex-start;margin:14px 0 6px}
.step .n{flex:none;width:26px;height:26px;border-radius:50%;background:#1d3a38;color:#63e0b8;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}
.step .t{font-weight:650;font-size:15px}
.preset{border:1px solid #263243;border-radius:12px;padding:10px 12px;cursor:pointer;background:#111821}
.preset:hover{border-color:#3a4d63}
.preset.sel{border-color:#2fb39f;background:#10221f}
.preset b{font-size:14px}
.presets{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;width:100%}
.note{border-radius:10px;padding:9px 12px;font-size:13px;margin-top:8px}
.note.warn{background:#2a2010;border:1px solid #5c4416;color:#f2c46a}
.note.bad{background:#2a1312;border:1px solid #6b2a26;color:#ff9b93}
.note.ok{background:#0f2a22;border:1px solid #1f5a47;color:#7fe0bd}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.spread{display:flex;justify-content:space-between;align-items:center;gap:10px;width:100%}
.job{display:flex;flex-direction:column;gap:8px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.lx .q-tab{font-size:15px;text-transform:none}
.lx .q-btn{text-transform:none;border-radius:10px}
.lx .q-field{width:100%}
.w-full{width:100%}
'''

PRESETS = [('check', 'Проверка', 50, 'проходит ли весь тракт'), ('short', 'Короткое', 5000, 'первый результат'),
           ('standard', 'Стандарт', 30000, 'рабочий навык'), ('full', 'Полное', 100000, 'максимум качества')]
STATUS = {'queued': ('В очереди', 'info'), 'running': ('Идёт', 'busy'), 'completed': ('Готово', 'ok'),
          'failed': ('Ошибка', 'bad'), 'cancelled': ('Отменено', ''), 'interrupted': ('Прервано', 'bad')}
NAMES = {'so101_pick_place': 'SO-101 · кубик в коробку', 'tank_dock': 'Tank · подъезд по камере',
         'tank_dock_rgbd_lidar': 'Tank · подъезд с глубиной и лидаром', 'rosmaster_dock': 'ROSMASTER · подъезд к цели',
         'rosmaster_pick_place': 'ROSMASTER · кубик в коробку',
         'rosmaster_dock_rgbd_lidar': 'ROSMASTER · подъезд с глубиной и лидарами',
         'rosmaster_pick_place_rgbd_lidar': 'ROSMASTER · кубик в коробку с глубиной и лидарами',
         'so101_pick_place_sim': 'SO-101 · кубик в коробку (симуляция)', 'tank_dock_sim': 'Tank · подъезд (симуляция)',
         'rosmaster_dock_sim': 'ROSMASTER · подъезд (симуляция)', 'rosmaster_pick_place_sim': 'ROSMASTER · кубик (симуляция)',
         'so101_handover_sim': 'SO-101 · передача человеку (симуляция = блок 4)', 'rosmaster_floor_pick_sim': 'ROSMASTER · взять с пола (симуляция = блок 3)'}


def task_name(name):
    return NAMES.get(name) or PROFILES.get(name, {}).get('label', name)


def chart_axes(y_type='value', **y):
    axis = {'axisLabel': {'color': '#8b98a8'}, 'axisLine': {'lineStyle': {'color': '#2a3646'}},
            'splitLine': {'lineStyle': {'color': '#1c2633'}}}
    return {'xAxis': {'type': 'value', **axis}, 'yAxis': {'type': y_type, **axis, **y}}


SIM_TASKS = [('tank', 'Tank · подъезд к цели', 'dock'), ('rosmaster', 'ROSMASTER · подъезд к цели', 'dock'),
             ('so101', 'SO-101 · кубик в коробку', 'arm')]


# ---------------------------------------------------------------- data helpers
def fmt_duration(seconds):
    if seconds is None:
        return '—'
    seconds = int(seconds)
    if seconds < 90:
        return f'{seconds} с'
    if seconds < 5400:
        return f'{seconds // 60} мин'
    return f'{seconds / 3600:.1f} ч'


def real_profiles():
    return {k: v for k, v in PROFILES.items() if v.get('domain', 'real') == 'real'}


def readiness(name):
    eps = datasets.eligible(name)
    groups = len(datasets.connected_groups(eps)) if eps else 0
    return {'episodes': len(eps), 'groups': groups, 'frames': sum(e['frames'] for e in eps), 'list': eps}


def big_number(text):
    m = re.fullmatch(r'([0-9.]+)([KMB]?)', text)
    if not m:
        return None
    return float(m.group(1)) * {'': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9}[m.group(2)]


def seconds_per_step():
    """Measured speed of the last real training on this machine (falls back to 2 s/step)."""
    for job in catalog.rows('jobs'):
        if job['status'] != 'completed':
            continue
        for log in sorted((DATA / 'jobs' / job['id']).glob('trainer-*.log')):
            found = re.findall(r'([0-9.]+)s/step', log.read_text(errors='replace')[-20000:])
            if found:
                return float(found[-1])
    return 2.0


def job_progress(job):
    cfg = json.loads(job['config'])
    total = cfg.get('steps') or (50 if cfg.get('smoke') else None)
    steps, losses, rate = 0, [], None
    for log in sorted((DATA / 'jobs' / job['id']).glob('trainer-*.log')):
        text = log.read_text(errors='replace')
        for s, loss in re.findall(r'step:([0-9.]+[KMB]?) .*?loss:([0-9.]+)', text):
            value = big_number(s)
            if value is not None:
                losses.append([value, float(loss)])
                steps = max(steps, value)
        found = re.findall(r'([0-9.]+)s/step', text[-4000:])
        rate = float(found[-1]) if found else rate
        bar = re.findall(r'\| *([0-9]+)/([0-9]+) \[', text[-4000:])
        if bar:
            steps = max(steps, int(bar[-1][0]))
            total = total or int(bar[-1][1])
    eta = (total - steps) * rate if (rate and total and job['status'] == 'running') else None
    return {'kind': cfg.get('kind', 'train'), 'steps': steps, 'total': total, 'losses': losses[-400:], 'eta': eta,
            'rate': rate, 'cfg': cfg}


def disk_free_gb():
    st = os.statvfs(DATA)
    return st.f_bavail * st.f_frsize / 1e9


# ---------------------------------------------------------------- simlab runner
class SimRunner:
    """Starts scripts/simlab_pipeline.py (train -> MuJoCo evaluation -> videos), one at a time,
    holding the shared compute lease. Runs started elsewhere are seen through simlab/active.json."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None

    def busy(self):
        return active_run() is not None

    def start(self, robot, kind, policy=None):
        with self.lock:
            if self.busy():
                raise RuntimeError('Уже идёт: ' + active_run()['title'])
            if any(j['status'] == 'running' for j in catalog.rows('jobs')):
                raise RuntimeError('GPU занят обучением на реальных показах; дождитесь его окончания')
            stamp = time.strftime('%m%d_%H%M%S')
            log = DATA / 'simlab' / 'logs' / f'{robot}-{kind}-{stamp}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, 'PYTHONPATH': str(ROOT), 'MUJOCO_GL': 'egl'}
            args = ['scripts/simlab_pipeline.py', robot, kind, '--log', str(log)]
            if policy:
                args += ['--policy', policy]
            handle = log.open('w')
            self.process = subprocess.Popen([str(ROOT / '.venv-sim/bin/python'), '-u', *args], cwd=ROOT, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            process = self.process

            def wait():
                try:
                    process.wait()
                finally:
                    handle.close()
                    catalog.event('info', f'Simlab {robot} {kind}: завершено (код {process.returncode})')
            threading.Thread(target=wait, daemon=True).start()
            time.sleep(.5)

    def stop(self):
        run = active_run()
        if run:
            try:
                os.killpg(os.getpgid(run['pid']), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


TITLES = {'check': 'проверочный прогон', 'full': 'полное обучение', 'clone': 'клонирование учителя', 'eval': 'оценка и видео'}


def active_run():
    path = DATA / 'simlab' / 'active.json'
    try:
        run = json.loads(path.read_text())
        os.kill(run['pid'], 0)
    except (OSError, ValueError, KeyError):
        return None
    run['title'] = dict((r, t) for r, t, _ in SIM_TASKS)[run['robot']] + ' · ' + TITLES.get(run['kind'], run['kind'])
    return run


def last_run(robot):
    """Most recent pipeline log of this robot (running or finished)."""
    logs = sorted((DATA / 'simlab' / 'logs').glob(f'{robot}-*.log'), key=lambda p: p.stat().st_mtime) if (DATA / 'simlab' / 'logs').exists() else []
    run = active_run()
    if run and run['robot'] == robot and run.get('log'):
        log = Path(run['log'])
    elif logs:
        log = logs[-1]
    else:
        return None
    rows, stage, done = [], None, False
    try:
        text = log.read_text(errors='replace')
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith('{"iter"'):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        elif line.startswith('{"stage"'):
            try:
                stage = json.loads(line)['stage']
                done = stage == 'готово'
            except ValueError:
                pass
    running = bool(run and run['robot'] == robot)
    kind = log.name.split('-')[1] if '-' in log.name else ''
    return {'rows': rows, 'stage': stage, 'running': running, 'done': done, 'kind': kind,
            'failed': not running and not done, 'started': run['started'] if running else log.stat().st_ctime,
            'tail': text[-1200:]}


RUNNER = SimRunner()


def sim_policies(robot):
    out = []
    root = DATA / 'simlab' / 'policies'
    for d in sorted(root.glob(f'{robot}_*'), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []:
        try:
            info = json.loads((d / 'policy.json').read_text())
        except (OSError, ValueError):
            continue
        evals = {}
        for f in d.glob('eval*.json'):
            try:
                evals[f.stem] = json.loads(f.read_text())['summary']
            except (OSError, ValueError, KeyError):
                pass
        out.append({'name': d.name, 'path': str(d), 'info': info, 'evals': evals, 'mtime': d.stat().st_mtime})
    return out


# ---------------------------------------------------------------- page
def install(jobs):
    from nicegui import app
    (DATA / 'simlab').mkdir(parents=True, exist_ok=True)
    app.add_media_files('/simlab-media', DATA / 'simlab')

    @ui.page('/learn')
    async def learn_page():
        ui.add_css(CSS)
        ui.dark_mode().enable()
        ui.colors(primary='#24a08f', secondary='#5b6b80', positive='#2fb39f', negative='#d9534f', warning='#f2b65a')
        state = {'profile': next(iter(real_profiles())), 'policy': 'act', 'preset': 'standard', 'source': 'new',
                 'excluded': set(), 'batch': 8, 'seed': 1000, 'device': 'cuda', 'custom_steps': None}

        with ui.element('div').classes('lx'):
            with ui.element('div').classes('lx-top'):
                with ui.column().style('gap:0'):
                    ui.label('Обучение').classes('lx-title')
                    ui.label('Реальные показы и симуляция MuJoCo · всё обучение Homebrain в одном месте').classes('lx-sub')
                chips = ui.element('div').classes('lx-chips')
            with ui.row().classes('row').style('margin:6px 0 14px'):
                ui.link('← Explorer', '/')
                ui.link('Сбор датасета', '/recording/collect')
                ui.link('Готовность парка', '/recording/readiness')

            @ui.refreshable
            def top_chips():
                active = [j for j in catalog.rows('jobs') if j['status'] == 'running']
                queued = [j for j in catalog.rows('jobs') if j['status'] == 'queued']
                if active:
                    ui.label('GPU: обучение ' + active[0]['profile']).classes('chip busy')
                elif active_run():
                    ui.label('GPU: ' + active_run()['title']).classes('chip busy')
                else:
                    ui.label('GPU свободен').classes('chip ok')
                if queued:
                    ui.label(f'В очереди: {len(queued)}').classes('chip info')
                free = disk_free_gb()
                ui.label(f'Диск: {free:.0f} ГБ свободно').classes('chip ' + ('bad' if free < 10 else 'ok' if free > 25 else 'busy'))
            with chips:
                top_chips()

            with ui.tabs().props('align=left inline-label no-caps active-color=positive indicator-color=positive') as tabs:
                t_real = ui.tab('real', 'Реальные показы', icon='videocam')
                t_sim = ui.tab('sim', 'Симуляция MuJoCo', icon='view_in_ar')
                t_jobs = ui.tab('jobs', 'Задания', icon='insights')
            with ui.tab_panels(tabs, value=t_real).classes('w-full').style('background:transparent'):
                with ui.tab_panel(t_real).style('padding:14px 0'):
                    real_tab(state, jobs, tabs)
                with ui.tab_panel(t_sim).style('padding:14px 0'):
                    sim_tab()
                with ui.tab_panel(t_jobs).style('padding:14px 0'):
                    jobs_tab(jobs, state, tabs)
            ui.timer(5, top_chips.refresh)

    # ------------------------------------------------------------ real demonstrations
    def real_tab(state, jobs, tabs):
        @ui.refreshable
        def task_cards():
            with ui.element('div').classes('grid'):
                for name, prof in real_profiles().items():
                    r = readiness(name)
                    pct = min(100, r['episodes'] * 2)
                    sel = ' sel' if name == state['profile'] else ''
                    card = ui.element('div').classes('card click' + sel)
                    card.on('click', lambda n=name: choose(n))
                    with card:
                        ui.html(f'<h3>{task_name(name)}</h3>')
                        with ui.element('div').classes('spread').style('margin:10px 0 6px'):
                            ui.html(f'<div><div class="num">{r["episodes"]}</div><div class="muted">успешных показов</div></div>')
                            ui.html(f'<div style="text-align:right"><div class="num">{r["groups"]}</div><div class="muted">серий-групп</div></div>')
                        ui.html(f'<div class="bar{"" if r["episodes"] >= 50 else " warn"}"><div style="width:{pct}%"></div></div>')
                        if r['groups'] < 3:
                            ui.label('Нужно ещё серий: минимум 3 разные').classes('muted').style('color:#ff9b93;margin-top:6px')
                        elif r['episodes'] < 50:
                            ui.label(f'До рекомендуемых 50: ещё {50 - r["episodes"]}').classes('muted').style('color:#f2c46a;margin-top:6px')
                        else:
                            ui.label('Данных достаточно для обучения').classes('muted').style('color:#7fe0bd;margin-top:6px')

        def choose(name):
            state['profile'] = name
            state['excluded'] = set()
            state['source'] = 'new'
            task_cards.refresh()
            settings.refresh()

        ui.label('1 · Выберите задачу').classes('lx-sub').style('font-size:15px;color:#c9d4e0;margin-bottom:8px')
        task_cards()

        @ui.refreshable
        def settings():
            name = state['profile']
            r = readiness(name)
            available = {}
            for ident, label in POLICIES.items():
                try:
                    validate_recipe(ident, name)
                    available[ident] = label
                except ValueError:
                    pass
            if state['policy'] not in available:
                state['policy'] = next(iter(available), 'act')
            sps = seconds_per_step()
            with ui.element('div').classes('grid2').style('margin-top:18px'):
                with ui.element('div').classes('card'):
                    ui.html('<div class="step"><div class="n">2</div><div><div class="t">Модель</div>'
                            '<div class="muted">ACT — основной рецепт LeRobot для ваших задач</div></div></div>')
                    ui.toggle(available, value=state['policy'],
                              on_change=lambda e: (state.update(policy=e.value), settings.refresh())).props('no-caps rounded unelevated toggle-color=positive')
                    sources = {'new': 'Новая модель'}
                    for j in catalog.rows('jobs'):
                        if (j['profile'] == name and j['policy'] == state['policy'] and j['status'] == 'completed'
                                and j['checkpoint'] and json.loads(j['config']).get('kind', 'train') == 'train'):
                            sources[j['id']] = 'Дообучить ' + j['id'][:8] + ' · ' + time.strftime('%d.%m %H:%M', time.localtime(j['created']))
                    if state['source'] not in sources:
                        state['source'] = 'new'
                    ui.select(sources, value=state['source'], label='Начать с',
                              on_change=lambda e: state.update(source=e.value)).style('margin-top:10px')

                    ui.html('<div class="step"><div class="n">3</div><div><div class="t">Сколько учить</div>'
                            f'<div class="muted">Оценка времени по скорости этой машины: {sps:.2f} с на шаг</div></div></div>')
                    with ui.element('div').classes('presets'):
                        for key, label, steps, hint in PRESETS:
                            sel = ' sel' if state['preset'] == key and not state['custom_steps'] else ''
                            p = ui.element('div').classes('preset' + sel)
                            p.on('click', lambda k=key: (state.update(preset=k, custom_steps=None), settings.refresh()))
                            with p:
                                ui.html(f'<b>{label}</b><div class="muted">{steps:,} шагов'.replace(',', ' ')
                                        + f'<br>≈ {fmt_duration(steps * sps)} · {hint}</div>')
                    with ui.expansion('Тонкие настройки', icon='tune').classes('w-full').style('margin-top:10px'):
                        ui.number('Своё число шагов', value=state['custom_steps'], min=50, max=1_000_000, step=50, precision=0,
                                  on_change=lambda e: state.update(custom_steps=int(e.value) if e.value else None))
                        ui.number('Размер пакета', value=state['batch'], min=1, max=32, precision=0,
                                  on_change=lambda e: state.update(batch=int(e.value or 8)))
                        ui.number('Seed', value=state['seed'], min=0, precision=0, on_change=lambda e: state.update(seed=int(e.value or 0)))
                        ui.select({'cuda': 'GPU', 'cpu': 'CPU'}, value=state['device'], label='Вычисления',
                                  on_change=lambda e: state.update(device=e.value))

                with ui.element('div').classes('card'):
                    used = [e for e in r['list'] if e['id'] not in state['excluded']]
                    groups = len(datasets.connected_groups(used)) if used else 0
                    ui.html('<div class="step"><div class="n">4</div><div><div class="t">Данные</div>'
                            '<div class="muted">Все успешные показы задачи. Серии делятся на обучение, проверку и итоговый тест целиком</div></div></div>')
                    with ui.element('div').classes('spread').style('margin:6px 0'):
                        ui.html(f'<div><div class="num">{len(used)}</div><div class="muted">показов</div></div>')
                        ui.html(f'<div><div class="num">{groups}</div><div class="muted">групп</div></div>')
                        ui.html(f'<div><div class="num">{sum(e["frames"] for e in used):,}</div><div class="muted">кадров</div></div>'.replace(',', ' '))
                    if groups < 3:
                        ui.html('<div class="note bad">Для честной проверки нужно минимум 3 независимые серии '
                                '(разные сессии и расстановки). Запишите ещё серии в «Сбор датасета».</div>')
                    elif len(used) < 50:
                        ui.html(f'<div class="note warn">Показов {len(used)} из рекомендуемых 50 — обучить можно, но навык будет слабее.</div>')
                    else:
                        ui.html('<div class="note ok">Данных достаточно.</div>')
                    with ui.expansion(f'Список показов ({len(r["list"])})', icon='list').classes('w-full').style('margin-top:8px'):
                        with ui.column().style('max-height:260px;overflow:auto;gap:2px;width:100%'):
                            for e in r['list']:
                                ui.checkbox(f"{e['id'][:8]} · {e['frames']} кадров · {e.get('session') or ''} {e.get('scene') or ''}",
                                            value=e['id'] not in state['excluded'],
                                            on_change=lambda ev, i=e['id']: (state['excluded'].discard(i) if ev.value else state['excluded'].add(i))).props('dense')
                    ui.html('<div class="step"><div class="n">5</div><div><div class="t">Запуск</div>'
                            '<div class="muted">Сначала проверка данных и показ плана, затем один клик</div></div></div>')
                    go = ui.button('Подготовить и запустить', icon='rocket_launch',
                                   on_click=lambda: prepare_and_start(go)).props('unelevated size=lg').classes('w-full')
                    go.set_enabled(groups >= 3)

        async def prepare_and_start(button):
            name = state['profile']
            steps = state['custom_steps'] or dict((k, s) for k, _, s, _ in PRESETS)[state['preset']]
            ids = [e['id'] for e in readiness(name)['list'] if e['id'] not in state['excluded']]
            button.props('loading')
            try:
                parent = None if state['source'] == 'new' else jobs.get(state['source'])
                draft_id, draft = await asyncio.to_thread(
                    training_drafts.prepare, name, state['policy'], ids, steps=int(steps), batch=int(state['batch']),
                    seed=int(state['seed']), device=state['device'], source=parent)
            except Exception as exc:
                ui.notify(str(exc), type='negative', multi_line=True)
                return
            finally:
                button.props(remove='loading')
            sps = seconds_per_step()
            with ui.dialog() as dlg, ui.card().style('background:#141b24;color:#e6edf3;min-width:min(520px,92vw);border-radius:16px'):
                ui.label('Готово к запуску').style('font-size:20px;font-weight:700')
                ui.label(f"{draft['policy'].upper()} · {task_name(name)}").classes('muted')
                names = {'train': 'Обучение', 'validation': 'Проверка во время обучения', 'test': 'Итоговый тест'}
                for role, count in draft['split_counts'].items():
                    with ui.element('div').classes('spread'):
                        ui.label(names[role])
                        ui.label(f"{count} показов · {draft['frames'][role]} кадров").classes('muted')
                ui.html(f"<div class='note ok'>{draft['steps']:,} шагов · пакет {draft['batch_size']} · "
                        f"примерно {fmt_duration(draft['steps'] * sps)}</div>".replace(',', ' '))
                with ui.row().style('justify-content:flex-end;width:100%;margin-top:10px'):
                    ui.button('Отмена', on_click=dlg.close).props('flat')

                    async def launch():
                        try:
                            ident = await asyncio.to_thread(jobs.start_prepared_training, draft_id)
                            ui.notify('Обучение поставлено в очередь · ' + ident[:8], type='positive')
                            dlg.close()
                            tabs.set_value('jobs')
                        except Exception as exc:
                            ui.notify(str(exc), type='negative', multi_line=True)
                    ui.button('Запустить', icon='play_arrow', on_click=launch).props('unelevated')
            dlg.open()

        settings()

    # ------------------------------------------------------------ simulation
    def sim_tab():
        ui.html('<div class="muted" style="margin-bottom:12px">Робот учится в MuJoCo на GPU: тысячи копий одновременно, '
                'с теми же датчиками, что у настоящего (лидар, гироскоп, метка цели). Каждый запуск сам проходит '
                '«обучение → проверка на 100 новых стартах в полной физике → видео». Допуск на робота — от 80 % успеха.</div>')

        @ui.refreshable
        def sim_cards():
            busy = RUNNER.busy()
            with ui.column().classes('w-full').style('gap:16px'):
                for robot, title, kind in SIM_TASKS:
                    pols = sim_policies(robot)
                    best = None
                    for p in pols:
                        for ev in p['evals'].values():
                            key = (ev.get('curriculum') == 1.0 or ev.get('curriculum') == 'easy (cube +-2 cm, as the teacher demos)', ev.get('success', 0))
                            if best is None or key > best[2]:
                                best = (p, ev, key)
                    show = best[0] if best else (pols[0] if pols else None)
                    run = last_run(robot)
                    with ui.element('div').classes('card job'):
                        with ui.element('div').classes('spread'):
                            ui.html(f'<h3 style="font-size:19px">{title}</h3>')
                            if best:
                                s_ = best[1]['success']
                                full = best[1].get('curriculum') == 1.0
                                ui.label(f'{s_ * 100:.0f} % успеха' + ('' if full else ' · облегчённые старты')).classes('chip ' + ('ok' if s_ >= .8 and full else 'busy'))
                            else:
                                ui.label('ещё не обучено').classes('chip')
                        if best:
                            ci = best[1].get('success_ci95')
                            ui.label(f"Лучшая политика {best[0]['name']} · проверка на {best[1].get('episodes')} новых стартах"
                                     + (f" · интервал {ci[0] * 100:.0f}–{ci[1] * 100:.0f} %" if ci else '')).classes('muted')
                        if run and (run['running'] or run['rows']):
                            rows = run['rows']
                            if run['running']:
                                last = rows[-1] if rows else None
                                ui.label(f"Идёт: {TITLES.get(run['kind'], run['kind'])} · этап «{run['stage'] or 'запуск'}» · {fmt_duration(time.time() - run['started'])}"
                                         + (f" · успех {last['success'] * 100:.0f} % · сложность {last['curriculum']:.1f}" if last else '')).classes('chip busy')
                            elif run['failed']:
                                ui.label('Последний запуск прерван').classes('chip bad')
                            if rows:
                                ui.echart({'backgroundColor': 'transparent', 'grid': {'left': 36, 'right': 10, 'top': 24, 'bottom': 22},
                                           **chart_axes(min=0, max=1), 'tooltip': {'trigger': 'axis'},
                                           'legend': {'textStyle': {'color': '#8b98a8'}, 'top': 0},
                                           'series': [{'type': 'line', 'showSymbol': False, 'name': 'успех', 'smooth': True,
                                                       'data': [[round(r['minutes'], 1), r['success']] for r in rows], 'lineStyle': {'color': '#63e0b8'}},
                                                      {'type': 'line', 'showSymbol': False, 'name': 'сложность стартов', 'step': 'end',
                                                       'data': [[round(r['minutes'], 1), r['curriculum']] for r in rows], 'lineStyle': {'color': '#79b8ff', 'type': 'dashed'}}]}
                                          ).style('height:160px;width:100%')
                        videos = []
                        if show:
                            try:
                                videos = json.loads((Path(show['path']) / 'videos' / 'index.json').read_text())
                            except (OSError, ValueError):
                                videos = []
                        if videos:
                            ui.label('Как это выглядит (новые старты, полная физика):').classes('muted')
                            with ui.element('div').classes('grid').style('grid-template-columns:repeat(auto-fill,minmax(280px,1fr))'):
                                for v in videos:
                                    with ui.column().style('gap:4px'):
                                        rel = Path(show['path']).relative_to(DATA / 'simlab') / 'videos' / v['file']
                                        player = ui.video(f'/simlab-media/{rel}', autoplay=False, loop=True, muted=True).style('width:100%;border-radius:10px')
                                        if v.get('poster'):
                                            player.props(f'poster="/simlab-media/{rel.parent / v["poster"]}"')
                                        label, cls = {'success': ('успех', 'ok'), 'collision': ('столкновение', 'bad'),
                                                      'timeout': ('не успел', 'busy')}.get(v['outcome'], (v['outcome'], 'bad'))
                                        ui.label(f"{label} · {v['seconds']} с").classes('chip ' + cls).style('align-self:flex-start')
                        with ui.row().classes('row'):
                            if kind == 'dock':
                                ui.button('Проверочный прогон · 10 мин', icon='bolt', on_click=lambda r=robot: run_(r, 'check')).props('unelevated').set_enabled(not busy)
                                ui.button('Полное обучение · ~4 ч', icon='rocket_launch', on_click=lambda r=robot: run_(r, 'full')).props('outline').set_enabled(not busy)
                            else:
                                ui.button('Клонировать учителя · ~15 мин', icon='school', on_click=lambda r=robot: run_(r, 'clone')).props('unelevated').set_enabled(not busy)
                            if show:
                                ui.button('Перепроверить и снять видео', icon='movie', on_click=lambda r=robot, p=show: run_(r, 'eval', p['path'])).props('flat').set_enabled(not busy)
                            if run and run['running']:
                                ui.button('Остановить', icon='stop', color='negative', on_click=lambda: (RUNNER.stop(), sim_cards.refresh())).props('flat')
                        if pols:
                            with ui.expansion(f'Все политики ({len(pols)})', icon='inventory_2').classes('w-full'):
                                for p in pols[:10]:
                                    evs = ' · '.join(f"сложн. {v.get('curriculum')}: {v['success'] * 100:.0f} %" for v in p['evals'].values()) or 'без оценки'
                                    ui.label(f"{p['name']} · {time.strftime('%d.%m %H:%M', time.localtime(p['mtime']))} · {evs}").classes('muted mono')

        def run_(robot, kind, policy=None):
            try:
                RUNNER.start(robot, kind, policy)
                ui.notify('Запущено: ' + TITLES[kind], type='positive')
            except Exception as exc:
                ui.notify(str(exc), type='negative')
            sim_cards.refresh()

        sim_cards()
        state = {'sig': None, 'at': time.time()}

        def maybe_refresh():
            # Rebuild on changes; while a run is active only every 30 s so videos are not interrupted.
            run = active_run()
            sig = (run and run['pid'], tuple(p['name'] + str(len(p['evals'])) for r, _, _ in SIM_TASKS for p in sim_policies(r)))
            if sig != state['sig'] or (run and time.time() - state['at'] > 30):
                state['sig'] = sig
                state['at'] = time.time()
                sim_cards.refresh()
        ui.timer(10, maybe_refresh)

    # ------------------------------------------------------------ jobs
    def jobs_tab(jobs, state, tabs):
        filt = {'value': 'train'}

        def show_log(ident):
            with ui.dialog() as dlg, ui.card().style('background:#0f141b;color:#e6edf3;width:min(1000px,94vw);max-width:94vw;border-radius:14px'):
                ui.label('Журнал ' + ident[:8]).style('font-weight:700')
                ui.code(jobs.log(ident), language='text').style('max-height:65vh;overflow:auto;width:100%')
                ui.button('Закрыть', on_click=dlg.close).props('flat')
            dlg.open()

        def act(action, ident):
            try:
                if action == 'cancel':
                    jobs.cancel(ident)
                elif action == 'resume':
                    jobs.resume(ident)
                elif action == 'evaluate':
                    jobs.evaluate(ident, 'validation')
                elif action == 'test':
                    jobs.evaluate(ident, 'test')
                elif action == 'finetune':
                    job = jobs.get(ident)
                    state.update(profile=job['profile'], policy=job['policy'], source=ident)
                    tabs.set_value('real')
                ui.notify({'cancel': 'Отменено', 'resume': 'Продолжение в очереди', 'evaluate': 'Проверка в очереди',
                           'test': 'Итоговый тест в очереди', 'finetune': 'Выбрано для дообучения'}[action], type='positive')
            except Exception as exc:
                ui.notify(str(exc), type='negative', multi_line=True)
            job_list.refresh()

        with ui.row().classes('row').style('margin-bottom:10px'):
            ui.toggle({'train': 'Обучение', 'all': 'Все задания'}, value='train',
                      on_change=lambda e: (filt.update(value=e.value), job_list.refresh())).props('no-caps rounded unelevated toggle-color=positive')

        @ui.refreshable
        def job_list():
            rows = catalog.rows('jobs')
            if filt['value'] == 'train':
                rows = [j for j in rows if json.loads(j['config']).get('kind', 'train') in ('train', 'evaluate')]
            if not rows:
                ui.label('Заданий пока нет. Начните на вкладке «Реальные показы».').classes('muted')
                return
            with ui.column().classes('w-full').style('gap:12px'):
                for job in rows[:40]:
                    p = job_progress(job)
                    label, cls = STATUS.get(job['status'], (job['status'], ''))
                    with ui.element('div').classes('card job'):
                        with ui.element('div').classes('spread'):
                            title = task_name(job['profile'])
                            kind = {'train': 'обучение', 'evaluate': 'проверка'}.get(p['kind'], p['kind'])
                            ui.html(f"<h3>{title} · {job['policy'].upper()} · {kind}</h3>")
                            ui.label(label).classes('chip ' + cls)
                        ui.label(f"{job['id'][:8]} · {time.strftime('%d.%m %H:%M', time.localtime(job['created']))}"
                                 + (f" · {p['cfg'].get('steps')} шагов" if p['cfg'].get('steps') else '')).classes('muted mono')
                        if p['total'] and p['kind'] == 'train':
                            pct = 100 * min(1, p['steps'] / p['total'])
                            ui.html(f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>')
                            ui.label(f"шаг {int(p['steps']):,} из {int(p['total']):,} · {pct:.0f} %".replace(',', ' ')
                                     + (f" · осталось ≈ {fmt_duration(p['eta'])}" if p['eta'] else '')
                                     + (f" · ошибка {p['losses'][-1][1]:.3f}" if p['losses'] else '')).classes('muted')
                        if len(p['losses']) > 2:
                            ui.echart({'backgroundColor': 'transparent', 'grid': {'left': 44, 'right': 10, 'top': 8, 'bottom': 22},
                                       **chart_axes('log'),
                                       'tooltip': {'trigger': 'axis'},
                                       'series': [{'type': 'line', 'showSymbol': False, 'smooth': True, 'data': p['losses'],
                                                   'lineStyle': {'color': '#63e0b8'}, 'areaStyle': {'color': 'rgba(99,224,184,.08)'}}]}
                                      ).style('height:130px;width:100%')
                        if job['error'] and job['status'] in ('failed', 'interrupted'):
                            ui.html(f"<div class='note bad'>{job['error'][:300]}</div>")
                        with ui.row().classes('row'):
                            ui.button('Журнал', icon='article', on_click=lambda i=job['id']: show_log(i)).props('flat dense')
                            if job['status'] in ('queued', 'running'):
                                ui.button('Отменить', icon='close', on_click=lambda i=job['id']: act('cancel', i)).props('flat dense color=negative')
                            if job['status'] in ('interrupted', 'failed', 'cancelled') and p['kind'] == 'train':
                                ui.button('Продолжить', icon='replay', on_click=lambda i=job['id']: act('resume', i)).props('flat dense')
                            if job['status'] == 'completed' and p['kind'] == 'train':
                                ui.button('Проверить на записях', icon='fact_check', on_click=lambda i=job['id']: act('evaluate', i)).props('flat dense')
                                ui.button('Итоговый тест', icon='verified', on_click=lambda i=job['id']: act('test', i)).props('flat dense')
                                ui.button('Дообучить', icon='trending_up', on_click=lambda i=job['id']: act('finetune', i)).props('flat dense')
                            ui.link('Подробно', '/jobs/' + job['id']).classes('muted')

        job_list()
        signature = {'value': None}

        def maybe_refresh():
            rows = catalog.rows('jobs')
            sig = tuple((j['id'], j['status']) for j in rows[:40])
            running = any(j['status'] == 'running' for j in rows)
            if sig != signature['value'] or running:
                signature['value'] = sig
                job_list.refresh()
        ui.timer(4, maybe_refresh)
