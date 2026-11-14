"""/swarm — one screen to run the fleet: live shared map (robots, paths, objects, SO-101 station,
floor seen by the cameras), the three robots' cameras and states, the running task's stages,
a readiness check and the command bar with STOP."""
import io
import math
import shutil
import threading
import time

import numpy as np

from .paths import DATA

NAMES = {'tank': 'Tank', 'rosmaster': 'ROSMASTER', 'so101': 'SO-101'}
COLORS = {'tank': (99, 224, 184), 'rosmaster': (242, 182, 90), 'so101': (140, 170, 255)}
STATES = {'mapping': 'строит карту', 'submap': 'строит свою карту', 'tracking': 'знает, где он', 'weak': 'неуверенно',
          'searching': 'ищет себя на карте', 'no_scan': 'лидар молчит'}
EXTRA_NAMES = {'door': 'дверь', 'table': 'стол', 'sofa': 'диван', 'bed': 'кровать', 'cabinet': 'шкаф', 'mug': 'кружка',
               'person': 'человек', 'cat': 'кошка', 'dog': 'собака', 'robot': 'робот', 'radiator': 'батарея', 'wheel': 'колесо'}
_cam_cache = {}
_cam_lock = threading.Lock()
_ready_cache = [0., {}]


def _font(size):
    from PIL import ImageFont
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_map(px_per_m=90):
    """PNG bytes of the shared map with everything the fleet knows, or None."""
    from PIL import Image, ImageDraw
    from .fleet_map import FLEET, RES, SIZE
    from . import world_memory, so101_handoff, explore
    from .scout import inventory
    with FLEET.lock:
        occ = FLEET.grid.occupied()
        free = FLEET.grid.known_free()
        seen = FLEET.grid.seen.copy()
        track = {r: dict(s) for r, s in FLEET.track.items()}
    ys, xs = np.nonzero(occ | free)
    if len(xs) == 0:
        return None
    x0, x1 = max(0, xs.min() - 16), min(SIZE, xs.max() + 17)
    y0, y1 = max(0, ys.min() - 16), min(SIZE, ys.max() + 17)
    img = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    img[:] = (16, 22, 30)
    img[free[y0:y1, x0:x1]] = (34, 47, 61)
    img[(free & seen)[y0:y1, x0:x1]] = (30, 62, 66)                # floor the cameras have looked at
    img[occ[y0:y1, x0:x1]] = (196, 212, 226)
    scale = px_per_m * RES
    im = Image.fromarray(img[::-1]).resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.NEAREST)
    d = ImageDraw.Draw(im)
    H = im.size[1]

    def P(x, y):
        return ((x / RES + SIZE // 2 - x0) * scale, H - (y / RES + SIZE // 2 - y0) * scale)

    small, big = _font(13), _font(15)
    # SO-101 station at the hand-over spot
    spot = so101_handoff.load().get('rosmaster_spot')
    if spot:
        sx, sy = P(spot[0], spot[1])
        d.rectangle([sx - 14, sy - 14, sx + 14, sy + 14], outline=COLORS['so101'], width=3)
        d.text((sx + 18, sy - 9), 'SO-101 · передача', fill=COLORS['so101'], font=small)
    # objects
    names = {**EXTRA_NAMES, **dict(inventory())}
    for item in world_memory.known_objects(max_age_s=24 * 3600):
        if not item.get('room_xy'):
            continue
        ox, oy = P(*item['room_xy'])
        age = time.time() - item['at']
        col = (255, 123, 114) if age < 600 else (170, 110, 105)
        d.line([ox - 6, oy - 6, ox + 6, oy + 6], fill=col, width=3)
        d.line([ox - 6, oy + 6, ox + 6, oy - 6], fill=col, width=3)
        d.text((ox + 8, oy - 8), names.get(item['prompt'], item['prompt']), fill=col, font=small)
    # robots, paths, targets
    for robot, s in track.items():
        col = COLORS.get(robot, (200, 200, 200))
        path = s.get('path')
        pose = s.get('pose')
        if path and pose:
            pts = [P(pose[0], pose[1])] + [P(*q) for q in path]
            d.line(pts, fill=col, width=2)
        goal = explore.ACTIVE.get(robot)
        if goal is not None and goal.target is not None and goal.frame == 'main':
            tx, ty = P(*goal.target)
            d.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], outline=col, width=2)
        if pose is None:
            continue
        cx, cy = P(pose[0], pose[1])
        r = .17 * px_per_m
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
        hx, hy = cx + math.cos(pose[2]) * r * 1.9, cy - math.sin(pose[2]) * r * 1.9
        d.line([cx, cy, hx, hy], fill=col, width=4)
        d.text((cx + r + 4, cy - r - 14), NAMES.get(robot, robot), fill=col, font=big)
    # scale bar
    d.line([14, H - 16, 14 + px_per_m, H - 16], fill=(200, 210, 220), width=3)
    d.text((18 + px_per_m, H - 24), '1 м', fill=(200, 210, 220), font=small)
    out = io.BytesIO()
    im.save(out, 'PNG')
    return out.getvalue()


def camera_jpeg(robot):
    """Latest camera frame of a robot as JPEG bytes (cached ~1 s), or None."""
    with _cam_lock:
        hit = _cam_cache.get(robot)
        if hit and time.monotonic() - hit[0] < .9:
            return hit[1]
    data = None
    try:
        if robot == 'rosmaster':
            from . import rosmaster_control
            with rosmaster_control._connection(3) as client:
                r = client.get('/frame.jpg')
                data = r.content if r.status_code == 200 else None
        elif robot == 'tank':
            # Tank's camera runs while it has a task (the swarm opens its session); the agent serves it.
            import httpx
            from .tank import config
            cfg = config()
            r = httpx.get(f"http://{cfg['host']}:8041/frame.jpg", headers={'X-Explorer-Token': cfg['token']}, timeout=3, trust_env=False)
            data = r.content if r.status_code == 200 and r.content else None
    except Exception:
        data = None
    with _cam_lock:
        _cam_cache[robot] = (time.monotonic(), data)
    return data


def readiness(control):
    """[(name, ok, note)] — what has to work before the fleet starts; cached 5 s."""
    if time.monotonic() - _ready_cache[0] < 5:
        return _ready_cache[1]
    from .fleet_map import FLEET, LOCALIZER
    from . import so101_handoff, rosmaster_control
    from .goto import Detector
    out = []
    st = FLEET.status()['robots']
    for robot in ('rosmaster', 'tank'):
        s = st.get(robot)
        fresh = s is not None and time.time() - s.get('at', 0) < 5
        out.append((f'{NAMES[robot]}: лидар', fresh and s.get('state') != 'no_scan',
                    STATES.get(s.get('state'), s.get('state')) if fresh else (LOCALIZER.errors.get(robot) or 'нет данных')[:60]))
    try:
        rosmaster_control.request('/health', timeout=4)
        out.append(('ROSMASTER: агент', True, 'на связи'))
    except Exception as exc:
        out.append(('ROSMASTER: агент', False, str(exc)[:60]))
    try:
        import httpx
        from .tank import config
        cfg = config()
        r = httpx.get(f"http://{cfg['host']}:8041/health", headers={'X-Explorer-Token': cfg['token']}, timeout=4, trust_env=False)
        out.append(('Tank: агент', r.status_code == 200, 'на связи' if r.status_code == 200 else f'ответ {r.status_code}'))
    except Exception as exc:
        out.append(('Tank: агент', False, str(exc)[:60]))
    miss = so101_handoff.missing()
    out.append(('Места передачи', not miss, 'запомнены' if not miss else f'не хватает: {len(miss)}'))
    det = Detector._instance
    out.append(('Распознавание SAM 3', det is not None and det.process.poll() is None,
                'загружено' if det is not None and det.process.poll() is None else 'загрузится при первой команде (~20 с)'))
    free_gb = shutil.disk_usage(DATA).free / 1e9
    out.append(('Диск', free_gb > 20, f'{free_gb:.0f} ГБ свободно'))
    _ready_cache[:] = [time.monotonic(), out]
    return out


def install(app, ui, control, notify_error, css=''):
    from fastapi.responses import Response

    @app.get('/api/swarm/map.png')
    def swarm_map():
        data = render_map()
        return Response(data, media_type='image/png', headers={'Cache-Control': 'no-store'}) if data else Response(status_code=204)

    @app.get('/api/swarm/camera/{robot}.jpg')
    def swarm_camera(robot: str):
        if robot not in ('tank', 'rosmaster'):
            return Response(status_code=404)
        data = camera_jpeg(robot)
        return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-store'}) if data else Response(status_code=204)

    @ui.page('/swarm')
    def page():
        from . import swarm as _swarm, world_memory, so101_handoff
        from .fleet_map import FLEET
        team = _swarm.SWARM
        ui.add_css(css + '''
        body{background:#0d1117}
        .sw-card{background:#141b24;border:1px solid #232f3e;border-radius:16px;padding:14px}
        .sw-chip{padding:3px 10px;border-radius:999px;font-size:12.5px;border:1px solid #2a3a4c;color:#c9d4e0;background:#111821}
        .sw-ok{color:#7fe0bd;border-color:#1f5a47;background:#0f2a22}
        .sw-bad{color:#ffb38a;border-color:#6b4226;background:#2a1a12}
        .sw-muted{color:#8b98a8;font-size:13px}
        .sw-stage{padding:6px 10px;border-radius:10px;background:#111821;border:1px solid #232f3e;font-size:13px}
        .sw-cam{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:10px;background:#0b0f15;border:1px solid #232f3e}
        ''')
        ui.dark_mode().enable()
        with ui.row().style('width:100%;align-items:center;justify-content:space-between;padding:10px 6px'):
            with ui.column().style('gap:2px'):
                ui.label('Рой Explorer').style('font-size:28px;font-weight:750;color:#e6edf3')
                ui.label('Tank · ROSMASTER · SO-101 — одна команда, общая карта и память').classes('sw-muted')
            with ui.row().style('gap:8px'):
                ui.link('Explorer', '/').style('color:#63e0b8')
                ui.link('Сбор датасета', '/recording/collect').style('color:#63e0b8')
                ui.link('Обучение', '/learn').style('color:#63e0b8')
                ui.button('STOP · все роботы', icon='stop', color='negative',
                          on_click=lambda: (team.stop('STOP со страницы роя'), control.stop('STOP со страницы роя'))).props('unelevated size=lg')
        ready_row = ui.row().style('gap:6px;flex-wrap:wrap;padding:0 6px 8px')
        with ui.element('div').classes('sw-card').style('margin:0 6px 12px'):
            with ui.row().style('width:100%;gap:8px;align-items:center;flex-wrap:wrap'):
                text = ui.input(placeholder='принеси печеньку · найдите носок · осмотрите комнату · подъедь к кубику').props('dense outlined dark').style('flex:1;min-width:320px')
                who = ui.select(['Оба робота', 'Только ROSMASTER', 'Только Tank'], value='Оба робота').props('dense outlined dark').style('min-width:170px')

                def go(value=None):
                    try:
                        v = (value or text.value or '').strip()
                        if not v:
                            raise ValueError('Напишите команду')
                        team.command(v, {'Оба робота': None, 'Только ROSMASTER': ['rosmaster'], 'Только Tank': ['tank']}[who.value])
                        ui.notify('Рой: ' + v)
                    except Exception as exc:
                        notify_error(exc)
                ui.button('Выполнить', icon='hub', on_click=lambda: go()).props('unelevated')
                for label in ('осмотрите комнату', 'найдите носок', 'принеси печенье'):
                    ui.button(label, on_click=lambda l=label: go(l)).props('flat dense no-caps')
                ui.button('Забрал', icon='back_hand', on_click=lambda: team.taken()).props('flat')
        with ui.row().style('width:100%;gap:12px;align-items:flex-start;flex-wrap:nowrap;padding:0 6px'):
            with ui.element('div').classes('sw-card').style('flex:3;min-width:0'):
                with ui.row().style('justify-content:space-between;width:100%;align-items:center'):
                    ui.label('Общая карта').style('font-weight:700;font-size:17px;color:#e6edf3')
                    map_note = ui.label('').classes('sw-muted')
                map_img = ui.image('/api/swarm/map.png').props('no-spinner no-transition').style('width:100%;border-radius:12px;background:#0b0f15')
                ui.label('Бирюзовый пол — его уже видели камеры (поиск идёт, пока не осмотрено всё). Кружок — цель робота, крестик — найденный предмет, квадрат — станция SO-101.').classes('sw-muted')
                with ui.row().style('gap:8px;margin-top:6px'):
                    ui.button('Начать карту заново', icon='restart_alt', on_click=lambda: (FLEET.reset(), ui.notify('Карта очищена'))).props('flat dense')
                    ui.button('Забыть предметы', icon='delete_sweep', on_click=lambda: (world_memory.forget(), ui.notify('Память очищена'))).props('flat dense')
            with ui.column().style('flex:2;min-width:300px;gap:12px'):
                cards = {}
                for robot in ('rosmaster', 'tank', 'so101'):
                    with ui.element('div').classes('sw-card').style('width:100%'):
                        with ui.row().style('justify-content:space-between;width:100%;align-items:center'):
                            ui.label(NAMES[robot]).style(f'font-weight:700;font-size:16px;color:rgb{COLORS[robot]}')
                            badge = ui.label('').classes('sw-chip')
                        if robot == 'so101':
                            cam = ui.image('').classes('sw-cam').props('no-spinner no-transition')
                        else:
                            cam = ui.image(f'/api/swarm/camera/{robot}.jpg').classes('sw-cam').props('no-spinner no-transition')
                        line1 = ui.label('').style('color:#e6edf3;font-size:13.5px')
                        line2 = ui.label('').classes('sw-muted')
                        cards[robot] = (badge, cam, line1, line2)
        with ui.element('div').classes('sw-card').style('margin:12px 6px;width:calc(100% - 12px)'):
            ui.label('Задание').style('font-weight:700;font-size:17px;color:#e6edf3')
            task_line = ui.label('Рой ждёт команду').style('color:#e6edf3')
            stages_row = ui.row().style('gap:6px;flex-wrap:wrap;margin:6px 0')
            log_box = ui.html('').classes('sw-muted')
        with ui.element('div').classes('sw-card').style('margin:0 6px 24px;width:calc(100% - 12px)'):
            ui.label('Что роботы нашли').style('font-weight:700;font-size:17px;color:#e6edf3')
            found_row = ui.row().style('gap:10px;flex-wrap:wrap')

        stage_names = {'search': 'поиск', 'explore': 'осмотр', 'approach': 'подъезд', 'pick': 'захват', 'deliver': 'доставка',
                       'place': 'на площадку', 'handoff': 'SO-101 передаёт', 'goto': 'подъезд'}
        last_found = [None]

        async def refresh():
            import asyncio
            t = time.time()
            checks = await asyncio.to_thread(readiness, control)
            map_img.set_source(f'/api/swarm/map.png?t={t:.1f}')
            st = FLEET.status()
            contrib = st.get('contrib', {})
            map_note.set_text(f"сканов: {st['scans']} · вклад: " + ', '.join(f"{NAMES.get(r, r)} {n}" for r, n in contrib.items()))
            ready_row.clear()
            with ready_row:
                for name, ok, note in checks:
                    ui.label(f'{name}: {note}').classes('sw-chip ' + ('sw-ok' if ok else 'sw-bad'))
            status = team.status()
            for robot, (badge, cam, line1, line2) in cards.items():
                if robot == 'so101':
                    live = control.robot == 'so101' and control.owner is not None
                    cam.set_source('/api/live/wrist.mjpg' if live else '')
                    cam.set_visibility(live)
                    h = team.handoff
                    badge.set_text('передаёт' if h is not None else ('на ведущей руке' if live else 'ждёт'))
                    line1.set_text('Станция передачи: ' + ('готова' if not so101_handoff.missing() else 'позы не показаны'))
                    line2.set_text('Камера запястья видна, когда SO-101 включён в Explorer')
                    continue
                cam.set_source(f'/api/swarm/camera/{robot}.jpg?t={t:.1f}')
                cam.set_visibility(True)
                s = st['robots'].get(robot) or {}
                badge.set_text(STATES.get(s.get('state'), 'нет данных'))
                pose = s.get('pose')
                line1.set_text((status.get('robots') or {}).get(robot) or s.get('message', ''))
                line2.set_text(f"({pose[0]:.2f}; {pose[1]:.2f}) м, курс {math.degrees(pose[2]):.0f}°" if pose else 'положение неизвестно')
                if robot == 'tank' and not _cam_cache.get('tank', (0, None))[1]:
                    cam.set_visibility(False)
                    line2.set_text(line2.text + ' · камера Tank включается, когда Tank в задании роя')
            task_line.set_text((f"«{status['task']}» · {status.get('message', '')}") if status.get('task') else 'Рой ждёт команду')
            stages_row.clear()
            with stages_row:
                seen_stages = []
                for at, stage, msg in status.get('log', []):
                    if stage in stage_names and stage not in seen_stages:
                        seen_stages.append(stage)
                for stage in seen_stages:
                    current = stage == status.get('stage') and status.get('active')
                    ui.label(('▶ ' if current else '✓ ') + stage_names[stage]).classes('sw-stage').style('border-color:#2fb39f' if current else '')
            log_box.set_content('<br>'.join(f"{time.strftime('%H:%M:%S', time.localtime(at))} · {msg}" for at, _, msg in status.get('log', [])[-6:]))
            items = world_memory.known_objects(max_age_s=24 * 3600)
            key = tuple((i['prompt'], i['at']) for i in items[:12])
            if key != last_found[0]:
                last_found[0] = key
                found_row.clear()
                from .scout import inventory
                names = {**EXTRA_NAMES, **dict(inventory())}
                with found_row:
                    if not items:
                        ui.label('Пока ничего: скажите «осмотрите комнату».').classes('sw-muted')
                    for i in items[:12]:
                        with ui.column().style('width:150px;gap:2px'):
                            if i.get('snapshot'):
                                ui.image('/world-media/snapshots/' + i['snapshot']).style('width:150px;border-radius:8px')
                            ui.label(names.get(i['prompt'], i['prompt'])).style('color:#e6edf3;font-weight:650')
                            ago = int((time.time() - i['at']) / 60)
                            where = f"({i['room_xy'][0]:.1f}; {i['room_xy'][1]:.1f})" if i.get('room_xy') else 'место неизвестно'
                            ui.label(f"{NAMES.get(i['robot'], i['robot'])}, {ago} мин назад · {where}").classes('sw-muted')
        ui.timer(1.0, refresh)
