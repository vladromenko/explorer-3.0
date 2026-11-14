"""Review an existing capture, with no new recording or robot connections."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
base='http://127.0.0.1:8030';out=Path('/home/jetsonnx/.local/share/explorer3/validation/ros2-hub')
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 deadline=time.monotonic()+30
 while True:
  try:records=page.request.get(base+'/api/fleet/captures').json()['records'];break
  except Exception:
   if time.monotonic()>deadline:raise
   time.sleep(.5)
 record=next(r for r in records if r.get('replay_frames'))
 page.goto(base+'/fleet/captures/'+record['id']);page.get_by_role('button',name='Вперёд',exact=True).wait_for()
 slider=page.locator('[role=slider]');initial=int(slider.get_attribute('aria-valuenow'))
 page.get_by_role('button',name='Вперёд',exact=True).click();page.wait_for_timeout(300)
 assert int(slider.get_attribute('aria-valuenow'))==initial+1
 page.get_by_role('button',name='Назад',exact=True).click();page.wait_for_timeout(300)
 assert int(slider.get_attribute('aria-valuenow'))==initial
 box=page.locator('.q-slider__track').bounding_box();page.mouse.click(box['x']+box['width']*.8,box['y']+box['height']/2);page.wait_for_timeout(500)
 assert int(slider.get_attribute('aria-valuenow'))>initial+2
 assert page.locator('img[src*="preview-"]').count()>=6
 page.get_by_role('button',name='Воспроизвести запись',exact=True).click();index=int(slider.get_attribute('aria-valuenow'));page.wait_for_timeout(1600)
 assert int(slider.get_attribute('aria-valuenow'))>index
 page.get_by_role('button',name='Пауза просмотра',exact=True).click();index=int(slider.get_attribute('aria-valuenow'));page.wait_for_timeout(1200)
 assert int(slider.get_attribute('aria-valuenow'))==index
 page.screenshot(path=str(out/'replay-desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200);page.screenshot(path=str(out/'replay-mobile.png'),full_page=True)
 assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
 assert not errors,errors
 result={'passed':True,'record':record['id'],'step_buttons':True,'slider_seek':True,'play_pause':True,'mobile':True,'page_errors':errors,'physical_commands':0}
 (out/'replay-browser.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));b.close()
