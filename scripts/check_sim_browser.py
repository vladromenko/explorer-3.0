"""Actual browser/actual physics acceptance; separate EXPLORER3_DATA server required."""
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright

ARTIFACTS=Path(os.environ.get('EXPLORER3_BROWSER_ARTIFACTS','/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser'))

async def main():
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={'width':1400,'height':1100})
        await context.tracing.start(screenshots=True,snapshots=True,sources=True)
        external=[]
        async def local_only(route):
            from urllib.parse import urlsplit
            host=urlsplit(route.request.url).hostname
            if host and host not in {'127.0.0.1','localhost'}:
                external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local_only)
        page=await context.new_page()
        errors=[]
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        try:
            await page.goto('http://127.0.0.1:8035/sim')
            await page.get_by_role('button',name='Открыть сцену',exact=True).click()
            frame=page.frame_locator('iframe')
            await frame.locator('canvas').first.wait_for(state='visible',timeout=60000)
            assert (await page.locator('iframe').bounding_box())['width']>500
            await page.get_by_role('button',name='Начать запись',exact=True).click()
            await page.get_by_text('Пауза · идёт запись',exact=True).wait_for(timeout=10000)
            await page.get_by_role('button',name='Двигать модель',exact=True).click()
            await page.get_by_text('Движение модели · идёт запись',exact=True).wait_for(timeout=15000)
            await page.wait_for_timeout(1200)
            await page.get_by_role('button',name='Пауза / STOP',exact=True).click()
            await page.get_by_text('Пауза · идёт запись',exact=True).wait_for(timeout=15000)
            await page.get_by_role('button',name='Закончить запись',exact=True).click()
            await page.get_by_text('Пауза · запись сохранена:',exact=False).wait_for(timeout=60000)
            await page.get_by_role('button',name='Сбросить сцену',exact=True).click()
            await page.screenshot(path=str(ARTIFACTS/'scene.png'),full_page=True)
            if errors or external:raise AssertionError({'errors':errors,'external':external})
            (ARTIFACTS/'result.json').write_text(json.dumps({'real_browser':True,'real_mujoco':True,
                'open_move_pause_reset':'PASS','record_save':'PASS','internet_requests':external,'console_errors':errors},indent=2))
        except Exception:
            await page.screenshot(path=str(ARTIFACTS/'failure.png'),full_page=True)
            (ARTIFACTS/'errors.json').write_text(json.dumps(errors,indent=2))
            raise
        finally:
            await context.tracing.stop(path=str(ARTIFACTS/'trace.zip'))
            await browser.close()

asyncio.run(main())
