"""Real UI finite fixture run, report/reopen and cancellation; no training."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    root=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-fleet-policies');root.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1100});await context.tracing.start(screenshots=True,snapshots=True)
        errors=[];external=[]
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local);page=await context.new_page();page.on('pageerror',lambda x:errors.append(str(x)))
        try:
            await page.goto('http://127.0.0.1:8035/sim/fleet/policies')
            await page.get_by_role('button',name='Исполнить политики · без обучения',exact=True).click()
            await page.get_by_text('Полное восстановление проверено',exact=False).wait_for(timeout=120000)
            async with page.expect_download() as download:
                await page.get_by_role('button',name='Скачать отчёт',exact=True).click()
            file=await download.value;await file.save_as(root/'report.json')
            report=json.loads((root/'report.json').read_text())
            assert report['replay']['verified'] and not report['training_started']
            assert len([e for e in report['events'] if e['kind']=='help_requested'])==2
            assert any(e['kind']=='help_started' for e in report['events'])
            assert set(report['policies'])=={'so101','tank'}
            assert report['correction_segments'] and all(s['frames']>0 for s in report['correction_segments'])
            await page.get_by_text('Программный пример не включается в обучающий набор',exact=False).first.wait_for()
            assert await page.get_by_role('button',name='Сохранить отдельное исправление',exact=True).count()==0
            await page.screenshot(path=str(root/'completed.png'),full_page=True)
            await page.reload();await page.get_by_label('Открыть сохранённый прогон',exact=True).click()
            await page.get_by_role('option').first.click()
            await page.get_by_text('Полное восстановление проверено',exact=False).wait_for(timeout=15000)
            await page.get_by_label('Время симуляции, с',exact=True).fill('20')
            await page.get_by_role('button',name='Исполнить политики · без обучения',exact=True).click()
            await page.get_by_text('Совместное исполнение',exact=False).last.wait_for(timeout=30000)
            await page.get_by_role('button',name='STOP',exact=True).click()
            await page.get_by_text('Остановлено',exact=False).wait_for(timeout=10000)
            assert not errors and not external
            (root/'result.json').write_text(json.dumps({'actual_browser':True,'shared_fixture_run':'PASS',
                'online_monitor_requests':2,'restore':report['replay'],'history_reopen':'PASS','cancel':'PASS',
                'training_started':False,'physical_robot_access':False,'errors':errors,'external':external},indent=2))
        except Exception:
            await page.screenshot(path=str(root/'failure.png'),full_page=True);raise
        finally:
            await context.tracing.stop(path=str(root/'trace.zip'));await browser.close()

asyncio.run(main())
