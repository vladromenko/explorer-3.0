#!/usr/bin/env python3
"""Read-only Tank browser acceptance using an isolated UI and separate diagnostic data.
Requires a reachable, operator-authorized Tank sensor observer; never acquires motors.
"""
import os,subprocess,tempfile,time,json,urllib.request,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1];real=Path(os.environ.get('EXPLORER3_DATA',str(Path.home()/'.local/share/explorer3')))
data=Path(tempfile.mkdtemp(prefix='explorer3-sensors-ui-'));(data/'configs').mkdir()
(data/'configs/workflow.json').write_text(json.dumps({'stage':'C','allowed_robots':['tank'],'hardware_connections':True,'training_authorized':False}))
shutil.copyfile(real/'configs/tank-agent.json',data/'configs/tank-agent.json');os.chmod(data/'configs/tank-agent.json',0o600)
(data/'configs/tank-hardware-enabled.json').write_text('{"enabled":true}')
env={**os.environ,'EXPLORER3_DATA':str(data),'EXPLORER3_PORT':'8035'}
with (data/'ui.log').open('w') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-m','explorer3.app'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for i in range(55):
   try:
    with urllib.request.urlopen('http://127.0.0.1:8035/api/health',timeout=1) as r:
     health=json.load(r);assert not health['armed']
     assert health['process_build']['pid']==p.pid, 'Test port belongs to another process'
     break
   except OSError:time.sleep(1)
  else:raise RuntimeError('UI did not start')
  from playwright.sync_api import sync_playwright
  with sync_playwright() as w:
   browser=w.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1420,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto('http://127.0.0.1:8035/tank/sensors');page.get_by_role('button',name='Подключить датчики',exact=True).click()
   page.get_by_text('Данные поступают',exact=True).wait_for(timeout=25000)
   page.get_by_role('button',name='Записать датчики · до 60 с',exact=True).click();page.wait_for_timeout(3500)
   page.get_by_role('button',name='Закончить запись',exact=True).click();page.get_by_text('Сохранено',exact=False).wait_for(timeout=10000)
   artifact=real/'validation/corrective-c/sensor-browser';artifact.mkdir(exist_ok=True)
   page.screenshot(path=str(artifact/'desktop.png'),full_page=True)
   assert page.locator('.q-img').count()==4
   assert all(page.locator('.q-img').nth(i).locator('img[src^="data:image/jpeg"]').count()>=1 for i in range(4))
   assert page.evaluate("Object.keys(window).filter(k=>k.startsWith('scene_c')).some(k=>window[k]?.children?.some(x=>x.isPoints && x.geometry.attributes.position.count>100))")
   page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(artifact/'mobile.png'),full_page=True)
   assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
   saved=list((data/'sensor-recordings').glob('*/manifest.json'))[0].parent.name
   page.goto('http://127.0.0.1:8035/tank/sensors?recording='+saved)
   page.get_by_text('Запись · пакет 1',exact=False).wait_for(timeout=12000)
   assert page.get_by_role('button',name='Подключить датчики',exact=True).count()==0
   assert page.locator('img[src^="data:image/jpeg"]').count()>=4
   assert not errors,errors
   browser.close()
  manifests=list((data/'sensor-recordings').glob('*/manifest.json'));assert len(manifests)==1
  m=json.loads(manifests[0].read_text());assert m['frames']>=3 and m['training_eligible'] is False and 'closed_at' in m
  result={'browser':'passed','errors':errors,'frames':m['frames'],'data':str(data),'read_only':True,'training_started':False}
  (artifact/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 finally:
  p.terminate()
  try:p.wait(timeout=25)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  print((data/'ui.log').read_text()[-2500:])
