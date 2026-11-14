"""Stationary real observation capture and download, no motor or training routes."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/home/jetsonnx/.local/share/explorer3/validation/ros2-hub');root.mkdir(parents=True,exist_ok=True)
base='http://127.0.0.1:8030'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 deadline=time.monotonic()+30
 while True:
  try:health=page.request.get(base+'/api/health').json();break
  except Exception:
   if time.monotonic()>deadline:raise
   time.sleep(.5)
 assert not health['armed'] and not health['rosmaster_manual_active']
 page.goto(base+'/fleet/live')
 page.get_by_text('Обе машины подключены · данные доступны вместе',exact=True).wait_for(timeout=30000)
 page.get_by_role('button',name='Записать датчики',exact=True).click()
 page.get_by_text('Записываем данные. Завершение автоматическое; управление роботами не меняется.',exact=True).wait_for(timeout=5000)
 deadline=time.monotonic()+140
 while time.monotonic()<deadline:
  state=page.request.get(base+'/api/fleet/captures').json()
  if state['records'] and not state['active']:
   record=state['records'][0];break
  page.wait_for_timeout(1000)
 else:raise AssertionError('Capture did not finish')
 assert record['phase']=='complete',(record['phase'],record.get('error'))
 assert set(record['observed_robots'])=={'tank','rosmaster'}
 assert record['readback']['passed'] and not record['training_eligible']
 page.get_by_text('Готово к записи доступных датчиков',exact=True).wait_for(timeout=5000)
 page.locator('.q-expansion-item').filter(has_text='сохранена').first.locator('.q-item').first.click()
 page.get_by_role('link',name='Отчёт о записи',exact=True).first.wait_for()
 href=page.get_by_role('link',name='Отчёт о записи',exact=True).first.get_attribute('href')
 assert page.request.get(base+href).json()['id']==record['id']
 baglinks=page.get_by_role('link',name='Скачать bag_',exact=False)
 assert baglinks.count()>0
 # Request the first saved file through the same UI download endpoint.
 response=page.request.get(base+baglinks.first.get_attribute('href'),timeout=30000)
 assert response.ok and response.body()[:8]==b'\x89MCAP0\r\n'
 assert record['replay_frames']>=10
 replay_link=page.get_by_role('link',name='Смотреть вместе',exact=True).first
 replay_link.click()
 page.get_by_text('Сохранённая сессия',exact=True).wait_for()
 # Seek through the actual widget and verify its value changes.
 slider=page.locator('[role=slider]');before=int(slider.get_attribute('aria-valuenow'))
 box=page.locator('.q-slider__track').bounding_box()
 page.mouse.click(box['x']+box['width']*.9,box['y']+box['height']/2);page.wait_for_timeout(700)
 assert int(slider.get_attribute('aria-valuenow'))>before
 assert page.locator('img[src*="preview-"]').count()>=6
 page.screenshot(path=str(root/'replay-desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
 assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
 page.screenshot(path=str(root/'replay-mobile.png'),full_page=True)
 page.goto(base+'/fleet/live');page.set_viewport_size({'width':1440,'height':1100})
 page.screenshot(path=str(root/'capture-desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
 page.screenshot(path=str(root/'capture-mobile.png'),full_page=True)
 assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
 assert not errors,errors
 after=page.request.get(base+'/api/health').json()
 assert not after['armed'] and not after['rosmaster_manual_active'] and not after['compute_job']
 result={'passed':True,'record_id':record['id'],'observed_robots':record['observed_robots'],
  'counts':record['readback']['counts'],'matched_clock_samples':record['matched_clock_samples'],
  'replay_frames':record['replay_frames'],'motion_commands_sent':0,'training_started':False,'page_errors':errors}
 (root/'capture-browser.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));browser.close()
