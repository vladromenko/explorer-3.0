"""ROSMASTER policy trial + help-pilot acceptance in a real browser, FAKE agent, isolated Explorer (no robot, no training).

PLAYWRIGHT_BROWSERS_PATH=~/.local/share/explorer3/cache/playwright .venv-sim/bin/python scripts/check_rosmaster_pilot_browser.py [checkpoint_dir]
Uses an existing smoke checkpoint (copied into a temporary EXPLORER3_DATA). The policy runs on CUDA inside the isolated UI and its
actions go only to the fake agent. Trials repeat until both pilot branches (help assigned / not assigned) were seen.
"""
import asyncio, json, os, secrets, shutil, signal, sqlite3, subprocess, sys, tempfile, time
from pathlib import Path
from playwright.async_api import async_playwright
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_rosmaster_browser import get, wait_http, ROOT

AGENT_PORT, UI_PORT = 8043, 8037
UI = f'http://127.0.0.1:{UI_PORT}'
DEFAULT_CKPT = Path.home() / '.local/share/explorer3/pipeline-check/20261112-190255/checkpoints/f50a3792648d463895dbaf4609763605-b8/checkpoints/last/pretrained_model'
SETUP = r'''
import hashlib, json, sys, time, uuid
from explorer3 import catalog, research
from explorer3.paths import DATA
from explorer3.profiles import profile
catalog.initialize(); research.initialize()
root = DATA / 'checkpoints' / 'fixture' / 'pretrained_model'
sha = hashlib.sha256((root / 'model.safetensors').read_bytes()).hexdigest()
cfg = profile('rosmaster_pick_place')
plan = {'id': uuid.uuid4().hex, 'kind': 'policy', 'profile': 'rosmaster_pick_place', 'source_id': 'fixture-smoke', 'fps': cfg['fps'],
        'channels': cfg['channels'], 'units': cfg['units'], 'max_duration_s': 60., 'physical_acceptance': False, 'motor_commands_sent': False,
        'state': 'prepared_requires_physical_block', 'created': time.time(), 'checkpoint': str(root), 'checkpoint_sha256': sha,
        'smoke_checkpoint': True, 'limitation': 'Smoke checkpoint is not a learned physical skill'}
(DATA / 'trials').mkdir(exist_ok=True); (DATA / 'trials' / (plan['id'] + '.json')).write_text(json.dumps(plan))
(DATA / 'configs' / ('trial-' + plan['id'] + '-authorized.json')).write_text(json.dumps(
    {'trial_id': plan['id'], 'profile': plan['profile'], 'authorized_at': time.time(), 'expires_at': time.time() + 3600, 'by': 'fixture'}))
study = research.create_study('rosmaster_pick_place', sha, delays=(1.,), correction_limit=4.)
research.accept_timing(study, 1.5, [1.])
print(json.dumps({'plan': plan['id'], 'study': study}))
'''


def packets(since=0):
    return [r for r in get(f'http://127.0.0.1:{AGENT_PORT}/_requests')[since:] if r['kind'] == 'packet']


async def run_trials(report, ids):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width': 1300, 'height': 1000})
        errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
        await page.goto(UI + '/rosmaster/control', wait_until='load'); await page.wait_for_timeout(3000)
        await page.click('text=Испытания и исследование')
        await page.wait_for_function(f"[...document.getElementById('tplan').options].some(o=>o.value=='{ids['plan']}') && [...document.getElementById('tstudy').options].some(o=>o.value=='{ids['study']}')", timeout=20000)
        report['plan_and_study_listed_in_panel'] = True
        trials, seen = [], set()
        for attempt in range(10):
            if {0, 1} <= seen and attempt >= 3: break
            await page.select_option('#tplan', ids['plan']); await page.select_option('#tstudy', ids['study'])
            await page.click('button:has-text("Запустить")')
            t = {'attempt': attempt}
            # Wait for the random decision (made 1.5 s after start); give model loading time on the first trial.
            assigned = None; end = time.time() + 60
            while time.time() < end:
                bar = await page.evaluate("[document.getElementById('pilotbar').style.display, document.getElementById('pilotbar').textContent]")
                if bar[0] == 'block' and 'ждёт помощи' in bar[1]: assigned = 1; t['hold_banner'] = bar[1]; break
                if bar[0] == 'block' and 'КОРРЕКЦИЯ' in bar[1]: assigned = 1; break
                dec = decisions(ids['data'])
                if len(dec) > len(trials) and not dec[-1]['assigned']: assigned = 0; break
                await page.wait_for_timeout(200)
            t['assigned'] = assigned
            if assigned == 1:
                await page.wait_for_function("document.getElementById('pilotbar').textContent.includes('ВАША КОРРЕКЦИЯ')", timeout=15000)
                t['correction_banner'] = await page.inner_text('#pilotbar')
                n = len(packets())
                await page.click('#video'); await page.keyboard.down('KeyJ'); await page.wait_for_timeout(700); await page.keyboard.up('KeyJ')
                during = packets(n)
                t['human_phase_packets_have_no_policy_action'] = bool(during) and all(not x['value']['direct'] for x in during[-5:])
                await page.click('button:has-text("Коррекция готова")'); await page.wait_for_timeout(1500)
                t['policy_resumed_after_correction'] = any(x['value']['source'] == 'policy' and x['value']['direct'] for x in packets(n))
            else:
                await page.wait_for_timeout(1500)
                t['policy_kept_control'] = any(x['value']['source'] == 'policy' and x['value']['direct'] for x in packets(len(packets()) - 10))
            if assigned is not None: seen.add(assigned)
            if attempt == 0 and assigned is not None:
                await page.click('button:has-text("Аварийное вмешательство")'); t['finished_with'] = 'emergency'
            else:
                await page.click('text=✓ Успех'); t['finished_with'] = 'success'
            await page.wait_for_function("document.getElementById('recstart').style.display!='none'", timeout=60000)
            await page.wait_for_timeout(1000)
            trials.append(t)
        report['trials'] = trials; report['both_branches_seen'] = {0, 1} <= seen
        report['page_errors'] = errors
        await page.screenshot(path=str(Path(ids['data']) / 'pilot.png'))
        await browser.close()


def decisions(data):
    db = sqlite3.connect(Path(data) / 'catalog.sqlite3'); db.row_factory = sqlite3.Row
    try: return [json.loads(r['record']) for r in db.execute('SELECT record FROM decisions ORDER BY created')]
    except sqlite3.OperationalError: return []


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    data = Path(tempfile.mkdtemp(prefix='explorer3-rosmaster-pilot-'))
    (data / 'configs').mkdir(parents=True)
    (data / 'configs/workflow.json').write_text(json.dumps({'stage': 'A', 'hardware_connections': False}))
    (data / 'configs/rosmaster-hardware-enabled.json').write_text(json.dumps({'enabled': True, 'scope': 'browser fixture'}))
    (data / 'configs/rosmaster-agent.json').write_text(json.dumps({'host': '127.0.0.1', 'port': AGENT_PORT, 'token': secrets.token_hex(8), 'manual_control_enabled': True}))
    shutil.copytree(ckpt, data / 'checkpoints/fixture/pretrained_model')
    env = {**os.environ, 'EXPLORER3_DATA': str(data), 'PYTHONPATH': str(ROOT)}
    py = str(ROOT / '.venv/bin/python')
    ids = json.loads(subprocess.check_output([py, '-c', SETUP], cwd=ROOT, env=env).decode().strip().splitlines()[-1]); ids['data'] = str(data)
    report = {'domain': 'fixture', 'robot_touched': False, 'training_started': False, 'data': str(data), 'checkpoint_source': str(ckpt)}
    agent = subprocess.Popen([py, str(ROOT / 'scripts/fake_rosmaster_agent.py'), str(AGENT_PORT)], cwd=ROOT)
    ui = subprocess.Popen([py, '-m', 'explorer3.app'], cwd=ROOT, env={**env, 'EXPLORER3_PORT': str(UI_PORT), 'EXPLORER3_HOST': '127.0.0.1'},
                          stdout=open(data / 'ui.log', 'w'), stderr=subprocess.STDOUT)
    try:
        wait_http(f'http://127.0.0.1:{AGENT_PORT}/health'); wait_http(UI + '/rosmaster/control')
        asyncio.run(run_trials(report, ids))
        time.sleep(3)
        db = sqlite3.connect(data / 'catalog.sqlite3'); db.row_factory = sqlite3.Row
        report['episodes'] = [dict(r) for r in db.execute('SELECT id,profile,category,outcome,frames FROM episodes ORDER BY rowid')]
        report['decisions'] = [{k: d.get(k) for k in ('episode_id', 'assigned', 'propensity', 'assigned_delay_s', 'outcome', 'emergency')} |
                               {'events': [e.get('kind') for e in d.get('events', [])]} for d in decisions(data)]
        segs = []
        for prov in sorted((data / 'datasets').glob('**/provenance.json')):
            v = json.loads(prov.read_text())
            h = v.get('hardware', {})
            segs.append({'trial_id': h.get('trial_id'), 'segment': h.get('segment_index'), 'of': h.get('segments'), 'source': h.get('command_source'),
                         'outcome': v.get('outcome'), 'termination_reason': h.get('termination_reason'), 'emergency': h.get('emergency_intervention')})
        report['segments'] = segs
    finally:
        for proc in (ui, agent):
            proc.send_signal(signal.SIGTERM)
            try: proc.wait(20)
            except subprocess.TimeoutExpired: proc.kill()
    out = ROOT / 'docs/handoffs/rosmaster-pilot-browser-acceptance.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
