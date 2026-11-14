"""End-to-end Homebrain acceptance. Runs simulation/inference, never a trainer."""
import json
import sys
import hashlib
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE='http://127.0.0.1:8030';DATA=Path('/home/jetsonnx/.local/share/explorer3');OUT=DATA/'validation/cognitive'


def jobs(page):
    return [j for j in page.request.get(BASE+'/api/status').json()['jobs'] if json.loads(j['config']).get('kind')=='cognitive_simulation']


def choose(page,label):
    page.get_by_label('Режим проверки').click()
    page.get_by_role('option',name=label,exact=True).click()


def launch(page):
    old={j['id'] for j in jobs(page)};page.get_by_role('button',name='Выполнить в симуляции',exact=True).click()
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        page.wait_for_timeout(200)
        new=[j for j in jobs(page) if j['id'] not in old]
        if new:return new[0]['id']
    raise AssertionError('UI did not enqueue a job')


def wait(page,ident,timeout=150):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        page.wait_for_timeout(500)
        job=next(j for j in jobs(page) if j['id']==ident)
        if job['status'] not in ('queued','running'):return job
    raise AssertionError('UI job deadline exceeded')


deadline=time.monotonic()+20
while True:
    try:
        with urllib.request.urlopen(BASE+'/api/health',timeout=1):break
    except OSError:
        if time.monotonic()>deadline:raise
        time.sleep(.2)

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100});errors=[];external=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    def guard(route):
        if route.request.url.startswith(('http://127.0.0.1:8030/','data:','blob:')):route.continue_()
        else:external.append(route.request.url);route.abort()
    page.route('**/*',guard);page.goto(BASE+'/team')
    runs=[]
    for mode,label in [('M1','С моделью, без памяти'),('M3','С моделью и проверяемой памятью')]:
        if (mode=='M1' and len(sys.argv)>1) or (mode=='M3' and len(sys.argv)>2):
            ident=sys.argv[1 if mode=='M1' else 2];job=next(j for j in jobs(page) if j['id']==ident)
            saved=json.loads((DATA/'jobs'/ident/'experiment.json').read_text())
            source=Path(__file__).resolve().parents[1]/'explorer3/sim/cooperation.py'
            assert saved['evaluator_version']==hashlib.sha256(source.read_bytes()).hexdigest()
        else:
            choose(page,label);ident=launch(page);job=wait(page,ident)
        assert job['status']=='completed',job
        result=json.loads((DATA/'jobs'/ident/'result.json').read_text())
        assert result['helper']==result['recipient']==result['task']=='success'
        assert result['physical_robot_access'] is False and result['training_started'] is False
        assert result['proposal']['backend']=='smolagents.ToolCallingAgent'
        assert result['proposal']['network_attempts']==[]
        assert result['proposal']['steps']<=2
        expected='scan_then_dock' if mode=='M1' else 'observe_then_dock'
        assert result['proposal']['proposal']['choice']==expected,result['proposal']
        runs.append({'id':ident,'mode':mode,'choice':expected,'model_seconds':result['proposal']['total_s'],
                     'ram_peak_kb':result['proposal']['rss_peak_kb'],'outcome':result['task']})
        page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT/'team-desktop.png'),full_page=True)
    page.get_by_role('tab',name='Помощь и результаты',exact=True).click()
    page.screenshot(path=str(OUT/'team-results.png'),full_page=True)
    page.get_by_role('tab',name='Общая память',exact=True).click()
    page.get_by_text('supported_in_scope',exact=False).first.wait_for()
    page.get_by_role('tab',name='Агенты',exact=True).click()
    assert page.get_by_text('Шесть ролей · одна локальная модель',exact=True).count()==1
    page.get_by_role('tab',name='Команда',exact=True).click()
    choose(page,'Фиксированный план · без модели')
    ident=launch(page);page.wait_for_timeout(2500)
    page.get_by_role('button',name='STOP · задача',exact=True).click()
    assert wait(page,ident,15)['status']=='cancelled'
    cancelled=[ident]
    page.wait_for_timeout(1200);ident=launch(page);page.wait_for_timeout(2500)
    # Exercise the installed visibilitychange listener; headless browsers do not
    # consistently mark a background tab hidden when another tab gains focus.
    page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'))")
    assert wait(page,ident,15)['status']=='cancelled';cancelled.append(ident)
    page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>false})")
    page.wait_for_timeout(1500)
    for ident in cancelled:
        assert not (DATA/'jobs'/ident/'result.json').exists()
        assert next(j for j in jobs(page) if j['id']==ident)['status']=='cancelled'
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
    page.screenshot(path=str(OUT/'team-mobile.png'),full_page=True)
    assert not errors,errors
    assert not external,external
    report={'passed':True,'real_browser':True,'real_mujoco':True,'real_local_model':True,'runs':runs,
        'stop_cancelled':cancelled[0],'visibility_listener_cancelled':cancelled[1],
        'visibility_fixture':'browser event injected because headless tab focus is not reliable',
        'page_errors':errors,'external_browser_requests':external,'mobile_overflow':False,
        'physical_commands':0,'optimizer_steps':0}
    (OUT/'team-browser.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    browser.close()
