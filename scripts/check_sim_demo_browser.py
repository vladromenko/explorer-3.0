"""Actual UI teacher demonstration and STOP; never presses a training button."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright


async def main():
    output=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-demo')
    output.mkdir(exist_ok=True)
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={'width':1400,'height':1100})
        await context.tracing.start(screenshots=True,snapshots=True)
        errors=[];external=[]
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:
                external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local)
        page=await context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            await page.goto('http://127.0.0.1:8035/sim')
            await page.get_by_role('button',name='Открыть сцену',exact=True).click()
            await page.frame_locator('iframe').locator('canvas').first.wait_for(timeout=60000)
            await page.locator('input[type=number]').fill('1')
            await page.locator('input[type=number]').press('Tab')
            async def state():return await (await context.request.get('http://127.0.0.1:8036/')).json()
            demo=page.get_by_role('button',name='Показать задачу · без обучения',exact=True)
            await demo.click();await page.wait_for_timeout(1500)
            assert (await state())['controller']=='scripted_expert'
            await page.get_by_role('button',name='Пауза / STOP',exact=True).click()
            await page.wait_for_timeout(500);stopped=await state();await page.wait_for_timeout(700)
            assert stopped['paused'] and stopped['frames']==(await state())['frames']
            await demo.click()
            await page.get_by_text('результат: success',exact=False).wait_for(timeout=60000)
            finished=await state()
            assert finished['paused'] and finished['task']['released'] and finished['task']['inside_box']
            await page.screenshot(path=str(output/'completed.png'),full_page=True)
            assert not errors and not external
            (output/'result.json').write_text(json.dumps({'actual_browser':True,'actual_mujoco':True,
                'scripted_hold_place_release':'PASS','stop_no_auto_resume':'PASS','training_started':False,
                'frames':finished['frames'],'model':finished['task']['model'],'errors':errors,'external':external},indent=2))
        except Exception:
            await page.screenshot(path=str(output/'failure.png'),full_page=True)
            raise
        finally:
            await context.tracing.stop(path=str(output/'trace.zip'));await browser.close()


asyncio.run(main())
