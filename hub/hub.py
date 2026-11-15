"""Homebrain «Пульт»: one page to start, stop and open the three robot systems.

  * Explorer 3.0            — Homebrain user services explorer3-ui (:8030) + explorer3-ros2-hub
  * Wristlab · SO-101       — Homebrain user services so101-ai-learning-ui (:8020) + MLflow (:5020)
  * Куб · ROSMASTER (CubeBot) — user service rosmaster-console on the robot (:8090)

Runs as its own always-on service (homebrain-hub, :8000), independent of the systems it
controls. Only fixed actions on fixed units: no shell input from the browser.
"""
import json
import socket
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

HERE = Path(__file__).resolve().parent
ROSMASTER = {'host': '192.168.0.177', 'user': 'vlad', 'key': str(Path.home() / '.ssh/rosmaster-collector')}

SYSTEMS = {
    'explorer': {'title': 'Explorer 3.0', 'subtitle': 'Tank, ROSMASTER и SO-101 вместе: управление, «подъехать к …», запись, обучение, симуляция',
                 'where': 'homebrain', 'units': ['explorer3-ui.service', 'explorer3-ros2-hub.service'], 'port': 8030,
                 'open': [('Рой', ':8030/swarm'), ('Открыть', ':8030/'), ('Сбор датасета', ':8030/recording/collect')]},
    'wristlab': {'title': 'Wristlab · SO-101', 'subtitle': 'Прежняя система обучения руки SO-101 и журнал экспериментов MLflow',
                 'where': 'homebrain', 'units': ['so101-ai-learning-ui.service', 'so101-ai-learning-mlflow.service'], 'port': 8020,
                 'open': [('Открыть', ':8020/'), ('MLflow', ':5020/')]},
    'cube': {'title': 'Куб · ROSMASTER', 'subtitle': 'CubeBot: найти жёлтый кубик, положить в коробку, достать (консоль на роботе)',
             'where': 'rosmaster', 'units': ['rosmaster-console.service'], 'port': 8090,
             'open': [('Открыть', 'robot:8090/')]},
}
NOTES = {
    'wristlab': 'SO-101 одновременно может управляться только одной системой. В Explorer это кнопки «Взять / Вернуть SO-101».',
    'cube': 'Пока работает Куб, не включайте управление ROSMASTER в Explorer: оба управляют одним роботом.',
}


def run(cmd, timeout=20):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def remote(cmd, timeout=20):
    return run(['ssh', '-i', ROSMASTER['key'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
                f"{ROSMASTER['user']}@{ROSMASTER['host']}", 'export XDG_RUNTIME_DIR=/run/user/1000; ' + cmd], timeout)


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=.6):
            return True
    except OSError:
        return False


def state(name):
    s = SYSTEMS[name]
    if s['where'] == 'homebrain':
        units = {u: run(['systemctl', '--user', 'is-active', u]).stdout.strip() for u in s['units']}
        up = port_open('127.0.0.1', s['port'])
    else:
        reachable = port_open(ROSMASTER['host'], 22)
        units = {}
        if reachable:
            r = remote('for u in ' + ' '.join(s['units']) + '; do systemctl --user is-active $u; done', 8)
            units = dict(zip(s['units'], r.stdout.split()))
        up = port_open(ROSMASTER['host'], s['port'])
        if not reachable:
            units = {u: 'робот не в сети' for u in s['units']}
    return {'name': name, **{k: s[k] for k in ('title', 'subtitle', 'port', 'open', 'where')}, 'units': units,
            'running': up, 'note': NOTES.get(name)}


def act(name, action):
    s = SYSTEMS[name]
    if action not in ('start', 'stop', 'restart'):
        raise HTTPException(400, 'Неизвестное действие')
    units = s['units'] if action != 'stop' else list(reversed(s['units']))
    if s['where'] == 'homebrain':
        r = run(['systemctl', '--user', action, *units], 60)
    else:
        if not port_open(ROSMASTER['host'], 22):
            raise HTTPException(503, 'ROSMASTER не в сети: включите робот')
        r = remote(f"systemctl --user {action} {' '.join(units)}", 60)
    if r.returncode:
        raise HTTPException(500, (r.stderr or r.stdout).strip()[:400] or 'Ошибка')
    return state(name)


app = FastAPI()


@app.get('/', response_class=HTMLResponse)
def index():
    return (HERE / 'index.html').read_text()


@app.get('/manifest.webmanifest')
def manifest():
    return JSONResponse({'name': 'Homebrain', 'short_name': 'Homebrain', 'start_url': '/', 'display': 'standalone',
                         'background_color': '#0d1117', 'theme_color': '#0d1117',
                         'icons': [{'src': '/icon.png', 'sizes': '512x512', 'type': 'image/png'}]},
                        media_type='application/manifest+json')


@app.get('/icon.png')
def icon():
    return FileResponse(HERE / 'icon.png')


@app.get('/api/state')
def all_states():
    return [state(n) for n in SYSTEMS]


@app.post('/api/{name}/{action}')
def do(name: str, action: str):
    if name not in SYSTEMS:
        raise HTTPException(404, 'Нет такой системы')
    return act(name, action)


# ---------------------------------------------------------------- Mac desktop launcher
MAC_LAUNCHER = r"""#!/bin/bash
# Роботы: меню запуска Explorer 3, Wristlab и Куба (ROSMASTER). Работает без Claude.
HB="${HOMEBRAIN:-192.168.0.168}"
ROBOT="192.168.0.177"
choice=$(osascript -e 'choose from list {"Explorer 3 · Рой", "Explorer 3 · главная", "Explorer 3 · сбор датасета", "Wristlab · SO-101", "Куб · ROSMASTER", "Пульт Homebrain"} with title "Роботы" with prompt "Что запустить?" default items {"Explorer 3 · Рой"}')
start() { curl -s -m 60 -X POST "http://$HB:8000/api/$1/start" >/dev/null || osascript -e 'display alert "Homebrain не отвечает" message "Проверьте, что Homebrain включён и Mac в той же сети."'; }
case "$choice" in
  "Explorer 3 · Рой") start explorer; sleep 2; open "http://$HB:8030/swarm" ;;
  "Explorer 3 · главная") start explorer; sleep 2; open "http://$HB:8030/" ;;
  "Explorer 3 · сбор датасета") start explorer; sleep 2; open "http://$HB:8030/recording/collect" ;;
  "Wristlab · SO-101") start wristlab; sleep 2; open "http://$HB:8020/" ;;
  "Куб · ROSMASTER") start cube; sleep 3; open "http://$ROBOT:8090/" ;;
  "Пульт Homebrain") open "http://$HB:8000/" ;;
esac
"""

MAC_INSTALL = """#!/bin/bash
# Installs the «Роботы» launcher on this Mac's Desktop.
set -e
dest="$HOME/Desktop/Роботы.command"
curl -fsS "http://192.168.0.168:8000/mac/launcher.command" -o "$dest"
chmod +x "$dest"
xattr -d com.apple.quarantine "$dest" 2>/dev/null || true
echo "Готово: на рабочем столе ярлык «Роботы». Дважды щёлкните его."
"""


@app.get('/mac/launcher.command', response_class=PlainTextResponse)
def mac_launcher():
    return MAC_LAUNCHER


@app.get('/mac/install.sh', response_class=PlainTextResponse)
def mac_install():
    return MAC_INSTALL
