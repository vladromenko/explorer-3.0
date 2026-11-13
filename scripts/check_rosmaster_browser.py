"""ROSMASTER panel acceptance in a real browser against a FAKE agent and an isolated Explorer (no robot).

Run with the sim venv (Playwright):  PLAYWRIGHT_BROWSERS_PATH=~/.local/share/explorer3/cache/playwright \
  .venv-sim/bin/python scripts/check_rosmaster_browser.py
Starts its own fake agent (port 8041) and Explorer UI (port 8036, temporary EXPLORER3_DATA) and stops only those.
"""
import asyncio, json, os, secrets, signal, subprocess, sys, tempfile, time
from pathlib import Path
import urllib.request
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
AGENT_PORT, UI_PORT = 8041, 8036
UI = f'http://127.0.0.1:{UI_PORT}'


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def wait_http(url, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2).read(); return
        except Exception:
            time.sleep(1)
    raise RuntimeError('not up: ' + url)


def kinds(since=0):
    return [r for r in get(f'http://127.0.0.1:{AGENT_PORT}/_requests')[since:]]


async def scenario(report):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width': 1300, 'height': 1000})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        await page.goto(UI + '/rosmaster/control', wait_until='load'); await page.wait_for_timeout(2500)
        report['1_cold_open_no_motor_open'] = not any(r['kind'] in ('acquire', 'session') for r in kinds())
        n = len(kinds())
        await page.click('text=▶ Я рядом — включить'); await page.wait_for_timeout(2500)
        got = [r['kind'] for r in kinds(n)]
        report['2_enable_acquires_and_opens_session'] = 'acquire' in got and 'session' in got
        await page.click('#video'); n = len(kinds())
        await page.keyboard.down('KeyW'); await page.wait_for_timeout(800)
        held = [r['value'] for r in kinds(n) if r['kind'] == 'command']
        await page.keyboard.up('KeyW'); n = len(kinds()); await page.wait_for_timeout(600)
        released = [r['value'] for r in kinds(n) if r['kind'] == 'command']
        report['3_hold_W_drives_forward'] = any(v and v.get('drive', [0])[0] > 0 for v in held)
        report['3_release_zero'] = bool(released) and all(v.get('drive', [0])[0] == 0 for v in released[-5:])
        n = len(kinds())
        await page.evaluate("Object.defineProperty(document,'hidden',{value:true,configurable:true});document.dispatchEvent(new Event('visibilitychange'))")
        await page.wait_for_timeout(1500)
        report['4_hidden_tab_stops'] = 'stop' in [r['kind'] for r in kinds(n)]
        await page.evaluate("Object.defineProperty(document,'hidden',{value:false,configurable:true})")
        await page.reload(wait_until='load'); await page.wait_for_timeout(2500)
        await page.select_option('#rprof', 'rosmaster_pick_place')
        await page.click('text=● Начать запись')
        await page.wait_for_function("document.getElementById('rstate').textContent.includes('кадров') && parseInt(document.getElementById('rstate').textContent.split('кадров ')[1]) >= 40", timeout=90000)
        await page.click('text=✓ Успех'); await page.wait_for_timeout(8000)
        report['5_record_state_text'] = await page.inner_text('#rstate')
        n = len(kinds())
        await page.click('button.stop'); await page.wait_for_timeout(1200)
        report['6_stop_button'] = 'stop' in [r['kind'] for r in kinds(n)]
        report['page_errors'] = errors
        await page.screenshot(path=str(report['_dir'] / 'rosmaster-panel.png'))
        await browser.close()


def main():
    data = Path(tempfile.mkdtemp(prefix='explorer3-rosmaster-browser-'))
    (data / 'configs').mkdir(parents=True)
    (data / 'configs/workflow.json').write_text(json.dumps({'stage': 'A', 'hardware_connections': False}))
    (data / 'configs/rosmaster-hardware-enabled.json').write_text(json.dumps({'enabled': True, 'scope': 'browser fixture'}))
    (data / 'configs/rosmaster-agent.json').write_text(json.dumps({'host': '127.0.0.1', 'port': AGENT_PORT, 'token': secrets.token_hex(8),
                                                                  'manual_control_enabled': True}))
    report = {'domain': 'fixture', 'robot_touched': False, 'data': str(data), '_dir': data}
    py = str(ROOT / '.venv/bin/python')
    agent = subprocess.Popen([py, str(ROOT / 'scripts/fake_rosmaster_agent.py'), str(AGENT_PORT)], cwd=ROOT)
    ui = subprocess.Popen([py, '-m', 'explorer3.app'], cwd=ROOT, env={**os.environ, 'EXPLORER3_DATA': str(data),
                          'EXPLORER3_PORT': str(UI_PORT), 'EXPLORER3_HOST': '127.0.0.1'}, stdout=open(data / 'ui.log', 'w'), stderr=subprocess.STDOUT)
    try:
        wait_http(f'http://127.0.0.1:{AGENT_PORT}/health'); wait_http(UI + '/rosmaster/control')
        asyncio.run(scenario(report))
        import sqlite3
        db = sqlite3.connect(data / 'catalog.sqlite3'); db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT id,profile,outcome,frames FROM episodes')]
        report['5_episode_saved'] = rows
    finally:
        for proc in (ui, agent):
            proc.send_signal(signal.SIGTERM)
            try: proc.wait(20)
            except subprocess.TimeoutExpired: proc.kill()
    report.pop('_dir')
    out = ROOT / 'docs/handoffs/rosmaster-browser-acceptance.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
