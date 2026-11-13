"""Real browser → finite branch job → results, with no optimizer or robot access."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    output=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-branches');output.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1000})
        await context.tracing.start(screenshots=True,snapshots=True)
        errors=[];external=[]
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:
                external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto('http://127.0.0.1:8035/sim/branches')
            await page.get_by_role('button',name='Сравнить три варианта',exact=True).click()
            await page.get_by_text('Сравнение завершено',exact=True).wait_for(timeout=120000)
            await page.get_by_text('Восстановление состояния подтверждено',exact=True).wait_for()
            table=await page.locator('table').inner_text()
            async with page.expect_download() as info:
                await page.get_by_role('button',name='Скачать полный отчёт',exact=True).click()
            download=await info.value;await download.save_as(output/'report.json')
            report=json.loads((output/'report.json').read_text())
            assert report['replay_verified'] and len(report['outcomes'])==6
            assert not report['training_started'] and report['policy']['kind']=='diagnostic_hold'
            assert not external and not errors
            await page.get_by_text('Проверил кадры и результат · сохранить как исправление в симуляции',exact=True).click()
            await page.get_by_role('button',name='Добавить исправление в датасеты',exact=True).click()
            await page.get_by_text('Исправление сохранено · проверка чтения PASS',exact=False).wait_for(timeout=120000)
            imported=await page.get_by_text('Исправление сохранено · проверка чтения PASS',exact=False).inner_text()
            await page.screenshot(path=str(output/'comparison.png'),full_page=True)
            (output/'result.json').write_text(json.dumps({'actual_browser':True,'lineage':report['lineage'],
                'table':table,'reviewed_import':imported,'dataset_context':'isolated software acceptance, scripted expert only','training_started':False,'physical_robot_access':False,'replay_verified':True,
                'errors':errors,'external':external},indent=2,ensure_ascii=False))
        except Exception:
            await page.screenshot(path=str(output/'failure.png'),full_page=True);raise
        finally:
            await context.tracing.stop(path=str(output/'trace.zip'));await browser.close()

asyncio.run(main())
