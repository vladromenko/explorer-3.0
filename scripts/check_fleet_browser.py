"""Actual same-app shared-world workflow; no training or hardware."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    output=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-fleet');output.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        ctx=await browser.new_context(viewport={'width':1440,'height':1100});await ctx.tracing.start(screenshots=True,snapshots=True)
        errors=[];external=[]
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'127.0.0.1','localhost'}:external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await ctx.route('**/*',local);page=await ctx.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        async def state():return await (await ctx.request.get('http://127.0.0.1:8040/')).json()
        async def wait_state(predicate,limit=100):
            for _ in range(limit):
                value=await state()
                if predicate(value):return value
                await page.wait_for_timeout(300)
            raise AssertionError('Fleet state timeout: '+json.dumps(value)[:500])
        try:
            await page.goto('http://127.0.0.1:8035/sim/fleet')
            await page.get_by_role('button',name='Открыть общий стенд',exact=True).click()
            await page.frame_locator('iframe').locator('canvas').first.wait_for(timeout=60000)
            await page.get_by_role('button',name='Показать три задачи · без обучения',exact=True).click()
            await wait_state(lambda s:s['frames']>25)
            await page.get_by_role('button',name='Сохранить состояние',exact=True).click()
            saved=await wait_state(lambda s:s['snapshot_available']);save_time=saved['world']['time_s']
            await wait_state(lambda s:s['world']['time_s']>save_time+1)
            await page.get_by_role('button',name='Вернуться к состоянию',exact=True).click()
            restored=await wait_state(lambda s:s['paused'] and s['world']['time_s']<=save_time+.1)
            await page.wait_for_timeout(600);assert (await state())['frames']==restored['frames']
            await page.get_by_role('button',name='Проверить очередь · без движения',exact=True).click()
            queued=await wait_state(lambda s:s.get('allocation'));assert queued['allocation']['concurrent_operators']==1
            await page.get_by_role('button',name='Показать три задачи · без обучения',exact=True).click()
            finished=await wait_state(lambda s:all(v['outcome']=='success' for v in s['world']['robots'].values()),180)
            assert finished['paused']
            await page.screenshot(path=str(output/'completed.png'),full_page=True)
            assert not errors and not external
            (output/'result.json').write_text(json.dumps({'actual_browser':True,'actual_shared_mujoco':True,
                'three_scripted_tasks':'PASS','snapshot_restore_paused':'PASS','single_operator_assignment':'PASS',
                'training_started':False,'physical_skill_verified':False,'errors':errors,'external':external,
                'model':finished['world']['model']},indent=2))
        except Exception:
            await page.screenshot(path=str(output/'failure.png'),full_page=True);raise
        finally:await ctx.tracing.stop(path=str(output/'trace.zip'));await browser.close()

asyncio.run(main())
