"""Upload a stored simulator frame, inspect with two real models, retain errors."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    output=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-vision');output.mkdir(exist_ok=True)
    source=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/laboratory/so101_pick_place_sim-transfer.png')
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        ctx=await browser.new_context(viewport={'width':1400,'height':1100});await ctx.tracing.start(screenshots=True,snapshots=True)
        errors=[];external=[]
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await ctx.route('**/*',local);page=await ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto('http://127.0.0.1:8035/sim/vision')
            await page.locator('input[type=file]').set_input_files(str(source))
            await page.get_by_text('Изображение загружено',exact=True).wait_for()
            await page.get_by_role('button',name='Проанализировать кадр',exact=True).click()
            await page.get_by_text('Анализ завершён',exact=True).wait_for(timeout=90000)
            await page.get_by_text('Найдено объектов:',exact=False).wait_for()
            yolo=await page.get_by_text('Найдено объектов:',exact=False).inner_text()
            await page.screenshot(path=str(output/'yoloe.png'),full_page=True)
            await page.get_by_text('YOLOE · быстрый поиск и маски',exact=True).click()
            await page.get_by_role('option',name='SAM 3 · выделение по описанию',exact=True).click()
            await page.get_by_role('button',name='Проанализировать кадр',exact=True).click()
            await page.get_by_text('Анализ завершён',exact=True).wait_for(state='hidden',timeout=5000)
            await page.get_by_text('Анализ завершён',exact=True).wait_for(timeout=90000)
            sam=await page.get_by_text('Найдено объектов:',exact=False).inner_text()
            await page.screenshot(path=str(output/'sam3.png'),full_page=True)
            assert not errors and not external
            (output/'result.json').write_text(json.dumps({'actual_browser':True,'yoloe':yolo,'sam3':sam,
                'image':str(source),'training_started':False,'motor_commands':False,'ground_truth':False,
                'errors':errors,'external':external},indent=2))
        except Exception:
            await page.screenshot(path=str(output/'failure.png'),full_page=True);raise
        finally:await ctx.tracing.stop(path=str(output/'trace.zip'));await browser.close()

asyncio.run(main())
