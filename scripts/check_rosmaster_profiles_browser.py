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
        await page.request.post(AGENT+'/_test/acquire-delay',data={'seconds':1.5})
        await page.get_by_role('button',name='▶ Я рядом — включить').click()
        await page.wait_for_timeout(250);await page.locator('button.stop.big').click()
        await page.wait_for_timeout(2000)
        assert not get(AGENT+'/health')['armed']
        assert not any(r['kind']=='session' for r in get(AGENT+'/_requests'))
        await page.request.post(AGENT+'/_test/acquire-delay',data={'seconds':0})
        checked=[]
        for profile in ('rosmaster_pick_place','rosmaster_dock'):
            await page.get_by_label('Что управляем').select_option(profile)
            await page.wait_for_timeout(300)
            n=len(get(AGENT+'/_requests'))
            # Safari-style activation: clicking a button need not move focus from an editor.
            await page.evaluate("$('rscene').focus();document.querySelector('button.go.big').click()")
            await page.wait_for_timeout(2500)
            assert await page.evaluate("document.activeElement.id!=='rscene'")
            sessions=[r['value'] for r in get(AGENT+'/_requests')[n:] if r['kind']=='session']
            assert sessions and sessions[-1]['profile']==profile,sessions
            assert await page.evaluate("document.activeElement.id!=='control-profile'")
            # Editing cannot drive, and a key held while leaving an editor must be released first.
            key='ArrowUp' if profile=='rosmaster_pick_place' else 'KeyW'
            await page.locator('#rscene').focus();n=len(get(AGENT+'/_requests'))
            await page.keyboard.down(key);await page.wait_for_timeout(150)
            assert await page.locator('#keyboardhint').is_visible()
            await page.evaluate("document.querySelector('button.go.big').click()")
            await page.wait_for_timeout(200)
            typed=[r['value'] for r in get(AGENT+'/_requests')[n:] if r['kind']=='command']
            assert typed and all(not any(p['drive']+p['arm']) for p in typed)
            await page.keyboard.up(key)
            # Inputs for the inactive component are ignored, not sent to end the session.
            start=len(get(AGENT+'/_requests'))
            key='KeyW' if profile=='rosmaster_pick_place' else 'ArrowUp'
            await page.keyboard.down(key);await page.wait_for_timeout(250);await page.keyboard.up(key)
            packets=[r['value'] for r in get(AGENT+'/_requests')[start:] if r['kind']=='command']
            assert packets
            channel='drive' if profile=='rosmaster_pick_place' else 'arm'
            assert all(not any(p[channel]) for p in packets)
            if profile=='rosmaster_pick_place':assert await page.locator('button.hold').first.is_disabled()
            start=len(get(AGENT+'/_requests'))
            key='ArrowUp' if profile=='rosmaster_pick_place' else 'KeyW'
            await page.keyboard.down(key);await page.wait_for_timeout(250);await page.keyboard.up(key)
            packets=[r['value'] for r in get(AGENT+'/_requests')[start:] if r['kind']=='command']
            channel='arm' if profile=='rosmaster_pick_place' else 'drive'
            assert any(p[channel][0]>0 for p in packets)
            await page.locator('button.stop.big').click();await page.wait_for_timeout(500);checked.append(profile)
        # STOP followed by explicit enable works repeatedly without page reload.
        for _ in range(3):
            n=len(get(AGENT+'/_requests'))
            await page.get_by_role('button',name='▶ Я рядом — включить').click()
            await page.wait_for_timeout(700)
            assert get(AGENT+'/health')['armed']
            await page.locator('button.stop.big').click();await page.wait_for_timeout(500)
            assert not get(AGENT+'/health')['armed']
            assert any(r['kind']=='stop' for r in get(AGENT+'/_requests')[n:])
        await page.get_by_role('button',name='▶ Я рядом — включить').click();await page.wait_for_timeout(700)
        await page.keyboard.down('KeyW');await page.wait_for_timeout(250)
        await page.locator('button.stop.big').click();await page.wait_for_timeout(500)
        n=len(get(AGENT+'/_requests'))
        await page.get_by_role('button',name='▶ Я рядом — включить').click();await page.wait_for_timeout(700)
        assert get(AGENT+'/health')['armed']
        assert all(not any(r['value']['drive']) for r in get(AGENT+'/_requests')[n:] if r['kind']=='command')
        await page.keyboard.up('KeyW')
        await page.locator('button.stop.big').click();await page.wait_for_timeout(500)
        await page.get_by_role('button',name='▶ Я рядом — включить').click();await page.wait_for_timeout(700)
        n=len([r for r in get(AGENT+'/_requests') if r['kind']=='session'])
        # Emulate a half-open transport: readyState stays OPEN, but data ceases.
        await page.evaluate('ws.onmessage=()=>{};ws.send=()=>{}')
        await page.wait_for_timeout(4000)
        assert not get(AGENT+'/health')['armed']
        assert len([r for r in get(AGENT+'/_requests') if r['kind']=='session'])==n
        await page.get_by_role('button',name='▶ Я рядом — включить').click();await page.wait_for_timeout(1000)
        assert get(AGENT+'/health')['armed']
        await page.locator('button.stop.big').click();await page.wait_for_timeout(500)
        # Render lifecycle states synchronously, without touching a real robot or creating a job.
        for profile in ('rosmaster_dock','rosmaster_pick_place','rosmaster_dock_rgbd_lidar','rosmaster_pick_place_rgbd_lidar'):
            await page.get_by_label('Что управляем').select_option('rosmaster_manual')
            await page.wait_for_timeout(400)
            await page.select_option('#rprof',profile)
            await page.evaluate("$('rscene').focus();document.querySelector('#recstart button').click()")
            await page.wait_for_function("document.getElementById('rstate').textContent.includes('Запись идёт')",timeout=90000)
            assert await page.evaluate("document.activeElement.id!=='rscene'")
            assert await page.locator('#rscene').is_disabled()
            assert await page.input_value('#control-profile')==profile.removesuffix('_rgbd_lidar')
            assert await page.locator('#control-profile').is_disabled()
            n=len(get(AGENT+'/_requests'))
            for key in ('KeyW','ArrowUp','KeyN'):
                await page.keyboard.down(key);await page.wait_for_timeout(250);await page.keyboard.up(key)
            packets=[r['value'] for r in get(AGENT+'/_requests')[n:] if r['kind']=='command']
            assert packets and get(AGENT+'/health')['armed']
            if profile.removesuffix('_rgbd_lidar')=='rosmaster_dock':
                assert all(not any(p['arm']) and not p['grip'] for p in packets)
                assert any(p['drive'][0]>0 for p in packets)
            else:
                assert all(not any(p['drive']) for p in packets)
                assert any(p['arm'][0]>0 for p in packets) and any(p['grip'] for p in packets)
            await page.get_by_role('button',name='✓ Успех').click()
            await page.wait_for_function("document.getElementById('rstate').textContent.startsWith('Сохранён показ')",timeout=90000)
            await page.locator('button.stop.big').click();await page.wait_for_timeout(500)
        assert not any(r['kind']=='profile_rejection' for r in get(AGENT+'/_requests'))
        # Lighting is independent of enable/heartbeat; after STOP it cannot restart motion.
        n=len(get(AGENT+'/_requests'))
        await page.get_by_text('Подсветка робота',exact=True).click()
        await page.get_by_role('button',name='Бирюзовый',exact=True).click()
        await page.wait_for_timeout(300)
        requests=get(AGENT+'/_requests')[n:]
        assert any(r['kind']=='rgb' and r['value']['rgb']==[0,70,50] for r in requests)
        assert not any(r['kind'] in ('acquire','session','command') for r in requests)
        assert not get(AGENT+'/health')['armed']
        states=await page.evaluate('''() => {
          const result=[];
          for(const recording of [{phase:'preparing',frames:0},{phase:'recording',frames:0},
              {phase:'recording',frames:4},{phase:'saving',frames:4},
              {phase:'idle',frames:4,episode:'abc12345',outcome:'success'},
              {phase:'idle',frames:0,error:'Показ не сохранён: нет кадров'}]) {
            render({battery:{},mirror:{},log:[],recording});
            result.push({text:$('rstate').textContent,disabled:[...$('recout').querySelectorAll('button')].map(b=>b.disabled)});
          }
          return result;
        }''')
        assert states[0]['disabled']==[True,True,False] and 'Пока не начинайте' in states[0]['text']
        assert states[1]['disabled']==[True,True,False]
        assert states[2]['disabled']==[False,False,False] and 'можно начинать' in states[2]['text']
        assert states[3]['disabled']==[True,True,True]
        assert 'Сохранён показ abc12345' in states[4]['text']
        assert 'не сохранён' in states[5]['text']
        assert not errors
        await browser.close()
        return {'domain':'fixture','physical_robot_access':False,'cold_open_no_acquire':True,'profiles':checked,
                'stop_reenable_cycles':3,'half_open_recovery_without_resume':True,'http_stop':True,
                'stop_cancels_pending_acquire_enable':True,'wrong_profile_inputs_ignored':True,
                'intended_inputs_received':True,'held_key_not_replayed_after_stop':True,
                'recording_preparation_and_saved_states':True,'recording_arm_base_profile_switches':True,
                'recorded_profiles':['rosmaster_dock','rosmaster_pick_place','rosmaster_dock_rgbd_lidar','rosmaster_pick_place_rgbd_lidar'],
                'lighting_after_stop_without_enable':True,
                'enable_and_record_release_editor_focus':True,'editor_held_key_not_replayed':True,'errors':errors}
def main():
    OUT.mkdir(exist_ok=True);data=Path(tempfile.mkdtemp(prefix='explorer3-profile-check-'));(data/'configs').mkdir()
    (data/'configs/workflow.json').write_text(json.dumps({'stage':'B','allowed_robots':['rosmaster']}))
    (data/'configs/local-only.json').write_text(json.dumps({'enabled':True}))
    (data/'configs/rosmaster-hardware-enabled.json').write_text(json.dumps({'enabled':True}))
    (data/'configs/rosmaster-agent.json').write_text(json.dumps({'host':'127.0.0.1','port':8050,'token':secrets.token_hex(8),'manual_control_enabled':True}))
    procs=[]
    try:
        for args,env,name in [([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/fake_rosmaster_agent.py'),'8050'],os.environ,'agent'),
                ([str(ROOT/'.venv/bin/python'),str(ROOT/'tests/rosmaster_sensor_fixture.py')],{**os.environ,'EXPLORER3_DATA':str(data),'PYTHONPATH':str(ROOT)},'sensors'),
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
