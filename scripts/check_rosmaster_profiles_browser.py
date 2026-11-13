"""Separate manual arm/base profiles, real browser and fake agent only."""
import asyncio,json,os,secrets,signal,subprocess,tempfile,time,urllib.request
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-b')
AGENT='http://127.0.0.1:8050';UI='http://127.0.0.1:8051'
def get(url):
    with urllib.request.urlopen(url,timeout=3) as r:return json.load(r)
def wait(url):
    end=time.monotonic()+40
    while time.monotonic()<end:
        try:get(url);return
        except Exception:time.sleep(.5)
    raise RuntimeError('Test server unavailable')
async def scenario():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox']);page=await browser.new_page();errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.goto(UI+'/rosmaster/control');await page.wait_for_timeout(1500)
        assert not any(r['kind'] in ('acquire','session') for r in get(AGENT+'/_requests'))
        checked=[]
        for profile in ('rosmaster_pick_place','rosmaster_dock'):
            await page.get_by_label('Что управляем').select_option(profile)
            await page.wait_for_timeout(300)
            n=len(get(AGENT+'/_requests'))
            await page.get_by_role('button',name='▶ Я рядом — включить').click()
            await page.wait_for_timeout(2500)
            sessions=[r['value'] for r in get(AGENT+'/_requests')[n:] if r['kind']=='session']
            assert sessions and sessions[-1]['profile']==profile,sessions
            await page.locator('button.stop.big').click();await page.wait_for_timeout(500);checked.append(profile)
        assert not errors
        await browser.close()
        return {'domain':'fixture','physical_robot_access':False,'cold_open_no_acquire':True,'profiles':checked,'errors':errors}
def main():
    OUT.mkdir(exist_ok=True);data=Path(tempfile.mkdtemp(prefix='explorer3-profile-check-'));(data/'configs').mkdir()
    (data/'configs/workflow.json').write_text(json.dumps({'stage':'B','allowed_robots':['rosmaster']}))
    (data/'configs/local-only.json').write_text(json.dumps({'enabled':True}))
    (data/'configs/rosmaster-hardware-enabled.json').write_text(json.dumps({'enabled':True}))
    (data/'configs/rosmaster-agent.json').write_text(json.dumps({'host':'127.0.0.1','port':8050,'token':secrets.token_hex(8),'manual_control_enabled':True}))
    procs=[]
    try:
        for args,env,name in [([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/fake_rosmaster_agent.py'),'8050'],os.environ,'agent'),
                ([str(ROOT/'.venv/bin/python'),'-m','explorer3.app'],{**os.environ,'EXPLORER3_DATA':str(data),'EXPLORER3_PORT':'8051','EXPLORER3_HOST':'127.0.0.1'},'ui')]:
            procs.append(subprocess.Popen(args,cwd=ROOT,env=env,stdout=open(data/(name+'.log'),'w'),stderr=subprocess.STDOUT))
        wait(AGENT+'/health');wait(UI+'/api/health')
        result=asyncio.run(scenario());(OUT/'profiles-browser.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        for proc in procs:
            proc.send_signal(signal.SIGTERM)
            try:proc.wait(15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()
