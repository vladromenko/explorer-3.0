"""Actual browser/actual physics acceptance; separate EXPLORER3_DATA server required."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright

parser=argparse.ArgumentParser();parser.add_argument('--profile',default='so101');args=parser.parse_args()
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
            names={'rosmaster':'ROSMASTER · рука','rosmaster_dock':'ROSMASTER · шасси','tank':'Tank · шасси'}
            if args.profile in names:
                await page.get_by_role('combobox').first.click()
                await page.get_by_role('option',name=names[args.profile],exact=True).click()
            await page.get_by_role('button',name='Открыть сцену',exact=True).click()
            frame=page.frame_locator('iframe')
            await frame.locator('canvas').first.wait_for(state='visible',timeout=60000)
            assert (await page.locator('iframe').bounding_box())['width']>500
            await page.get_by_role('button',name='Сбросить сцену',exact=True).click()
            await page.get_by_role('combobox').nth(1).click()
            await page.get_by_role('combobox').nth(1).press('End')
            await page.get_by_role('combobox').nth(1).press('Enter')
            await page.wait_for_timeout(800)
            varied_response=await context.request.get('http://127.0.0.1:8036/')
            assert (await varied_response.json())['task']['variation']['mode']=='varied'
            await page.get_by_role('button',name='Начать запись',exact=True).click()
            await page.get_by_text('Пауза · идёт запись',exact=True).wait_for(timeout=10000)
            async def state():
                response=await context.request.get('http://127.0.0.1:8036/')
                return await response.json()
            if args.profile in {'tank','rosmaster_dock'}:
                # A second browser must not steal the first operator's lease.
                other_context=await browser.new_context(viewport={'width':1400,'height':1100})
                await other_context.route('**/*',local_only)
                second=await other_context.new_page()
                await second.goto('http://127.0.0.1:8035/sim')
                await second.get_by_role('button',name='Открыть сцену',exact=True).click()
                await second.get_by_role('button',name='Двигать модель',exact=True).click()
                await second.get_by_text('Сценой управляет другая страница.',exact=False).wait_for(timeout=10000)
                await other_context.close()
                await page.wait_for_timeout(250)
                assert (await state())['paused']
            await page.get_by_role('button',name='Двигать модель',exact=True).click()
            await page.get_by_text('Движение модели · идёт запись',exact=True).wait_for(timeout=15000)
            if args.profile in {'tank','rosmaster_dock'}:
                forward=page.get_by_role('button',name='↑',exact=True)
                await forward.dispatch_event('pointerdown')
                await page.wait_for_timeout(1200)
                moving=await state()
                assert moving['target'][0]>0 and moving['frames']>5
                assert moving['task']['distance_m']<.85, moving['task']
                await forward.dispatch_event('pointerup')
                await page.wait_for_timeout(300)
                assert not any((await state())['target'])
                await page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                # Headless focus does not reliably emulate OS tab visibility: send the
                # real page lifecycle event that also fires when leaving the page.
                await page.evaluate("window.dispatchEvent(new Event('pagehide'))")
                await page.wait_for_timeout(1000)
                hidden=await state();assert hidden['paused'] and hidden['owner'] is None
                frames=hidden['frames'];await page.wait_for_timeout(400)
                assert (await state())['frames']==frames
            else:await page.wait_for_timeout(1200)
            await page.get_by_role('button',name='Пауза / STOP',exact=True).click()
            await page.get_by_text('Пауза · идёт запись',exact=True).wait_for(timeout=15000)
            await page.get_by_role('button',name='Закончить запись',exact=True).click()
            await page.get_by_text('Пауза · запись сохранена:',exact=False).wait_for(timeout=60000)
            await page.get_by_role('button',name='Сбросить сцену',exact=True).click()
            await page.get_by_role('combobox').nth(1).click()
            await page.get_by_role('combobox').nth(1).press('Home')
            await page.get_by_role('combobox').nth(1).press('Enter')
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(ARTIFACTS/'scene.png'),full_page=True)
            if errors or external:raise AssertionError({'errors':errors,'external':external})
            (ARTIFACTS/'result.json').write_text(json.dumps({'real_browser':True,'real_mujoco':True,'profile':args.profile,
                'open_move_pause_reset':'PASS','record_save':'PASS','variation_switch':'PASS','training_started':False,
                'internet_requests':external,'console_errors':errors},indent=2))
        except Exception:
            await page.screenshot(path=str(ARTIFACTS/'failure.png'),full_page=True)
            (ARTIFACTS/'errors.json').write_text(json.dumps(errors,indent=2))
            raise
        finally:
            await context.tracing.stop(path=str(ARTIFACTS/'trace.zip'))
            await browser.close()

asyncio.run(main())
