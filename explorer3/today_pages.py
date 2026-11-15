"""/today — «Что делать сегодня»: one screen that reads the real state (robots, disk, recorded
blocks, trainings and gates, experiments) and names the next step with a link. Rules, not a
language model: every number comes from the catalogue, the gate reports and the journals."""
import json
import shutil
import time

from .paths import DATA

TARGET = 50


def state():
    from . import catalog, research_log
    from .datasets import eligible
    from .deploy_gate import latest
    s = {}
    s['disk_gb'] = round(shutil.disk_usage(DATA).free / 1e9, 1)
    s['blocks'] = {name: len(eligible(name)) for name in ('rosmaster_pick_place', 'so101_pick_place', 'rosmaster_dock_rgbd_lidar', 'tank_dock_rgbd_lidar')}
    jobs = catalog.rows('jobs')
    s['running'] = [j for j in jobs if j['status'] in ('queued', 'running')]
    trained = {}
    for j in jobs:
        cfg = json.loads(j['config'])
        from pathlib import Path
        if (j['status'] == 'completed' and j['checkpoint'] and Path(j['checkpoint']).is_dir()
                and cfg.get('kind', 'train') == 'train' and not cfg.get('smoke')):
            key = (j['profile'], j['policy'])
            if key not in trained or j['created'] > trained[key]['created']:
                trained[key] = {**j, 'gate': latest(j['id'])}
    s['trained'] = trained
    rows = research_log.rows()
    s['trials'] = {c: sum(1 for r in rows if r['condition'] == c and r['world'] == 'real') for c in ('A1', 'A2', 'A3')}
    s['unconfirmed'] = sum(1 for r in rows if not r['confirmed'])
    return s


def steps(s):
    """Ordered (state, text, link) — 'done' / 'next' / 'later' / 'blocked'."""
    out = []
    if s['disk_gb'] < 10:
        out.append(('blocked', f'Мало места на диске: {s["disk_gb"]} ГБ (запись запрещена ниже 8 ГБ).', '/recording/readiness'))
    for name, label, block in (('rosmaster_pick_place', 'ROSMASTER · взять с пола', 3), ('so101_pick_place', 'SO-101 · передача человеку', 4)):
        n = s['blocks'][name]
        out.append(('done' if n >= TARGET else 'next', f'Блок {block} «{label}»: {n} из {TARGET} годных показов этого навыка', '/recording/collect'))
    for name, sim, label in (('rosmaster_pick_place', 'rosmaster_floor_pick_sim', 'ROSMASTER'), ('so101_pick_place', 'so101_handover_sim', 'SO-101')):
        sim_done = any(k[0] == sim for k in s['trained'])
        real = {k[1]: v for k, v in s['trained'].items() if k[0] == name}
        if not real:
            out.append(('next' if s['blocks'][name] >= 30 else 'later',
                        f'{label}: обучение на ваших показах ещё не запускалось' + (' (веса MuJoCo готовы)' if sim_done else ''), '/'))
            continue
        for policy, j in real.items():
            g = j['gate']
            if g is None:
                out.append(('next', f'{label} {policy.upper()}: обучено — нажмите «Допуск к роботу»', '/'))
            elif g['verdict'] == 'ready':
                out.append(('done', f'{label} {policy.upper()}: допущена — проведите 15 испытаний', '/'))
            else:
                out.append(('next', f'{label} {policy.upper()}: не допущена — {g["meaning"][:140]}', '/'))
    t = s['trials']
    pending = min(t.values()) < 10
    out.append(('next' if pending else 'done', f'Эксперимент поиска (реальные попытки): A1 {t["A1"]}, A2 {t["A2"]}, A3 {t["A3"]} из 10 пилотных', '/swarm'))
    if s['unconfirmed']:
        out.append(('next', f'Подтвердите результат {s["unconfirmed"]} попыток поиска (верно / ошибка / не найдено)', '/swarm'))
    if s['running']:
        out.append(('done', 'Сейчас выполняется: ' + ', '.join(f'{j["profile"]} {j["policy"]} ({j["status"]})' for j in s['running']), '/'))
    return out


def install(app, ui):
    @ui.page('/today')
    def page():
        ui.dark_mode().enable()
        ui.add_css('body{background:#0d1117} .t-card{background:#141b24;border:1px solid #232f3e;border-radius:14px;padding:14px;margin:8px 0}')
        ui.label('Что делать сегодня').style('font-size:28px;font-weight:750;color:#e6edf3;margin:10px 6px 0')
        ui.label('Следующий шаг по реальному состоянию системы. Числа — из каталога, отчётов допуска и журналов.').style('color:#8b98a8;margin:0 6px')
        box = ui.column().style('width:100%;max-width:980px;padding:0 6px')

        def refresh():
            s = state()
            box.clear()
            colors = {'done': '#7fe0bd', 'next': '#ffcf8a', 'later': '#8b98a8', 'blocked': '#ff8a80'}
            marks = {'done': '✓', 'next': '▶', 'later': '·', 'blocked': '■'}
            first = True
            with box:
                for st, text, link in steps(s):
                    with ui.element('div').classes('t-card').style('border-color:' + ('#2fb39f' if st == 'next' and first else '#232f3e')):
                        with ui.row().style('justify-content:space-between;width:100%;align-items:center'):
                            ui.label(f'{marks[st]} {text}').style(f'color:{colors[st]};font-size:15px')
                            ui.link('открыть', link).style('color:#63e0b8')
                    if st == 'next':
                        first = False
                ui.label('Подробно: docs/TOMORROW_RU.md и docs/ARTICLE_TASKS_RU.md · обновлено ' + time.strftime('%H:%M:%S')).style('color:#8b98a8;font-size:12px')
        refresh()
        ui.timer(10, refresh)
