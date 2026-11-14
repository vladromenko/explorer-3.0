"""Choose actual isolated sim records; prepare/reopen draft, NEVER click start."""
import asyncio,json,sqlite3
from pathlib import Path
from playwright.async_api import async_playwright,expect

DATA=Path('/home/jetsonnx/.local/share/explorer3/validation/browser-sim')

def jobs():
    with sqlite3.connect(DATA/'catalog.sqlite3') as db:return db.execute('select id,status,config from jobs order by id').fetchall()

async def main():
    root=Path('/home/jetsonnx/.local/share/explorer3/validation/corrective-a/browser-training-draft');root.mkdir(exist_ok=True)
    before=jobs()
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
            await page.goto('http://127.0.0.1:8035/sim/training')
            await expect(page.get_by_role('button',name='Запустить подготовленное обучение',exact=True)).to_be_disabled()
            await page.get_by_label('Задача',exact=True).click();await page.get_by_role('option',name='Tank · подъезд к цели',exact=True).click()
            await page.get_by_label('Показы и исправления для нового плана',exact=True).click()
            await expect(page.get_by_role('option')).to_have_count(3)
            options=await page.get_by_role('option').all();assert len(options)==3,await page.get_by_role('option').all_inner_texts()
            for option in options:await option.click()
            await page.keyboard.press('Escape')
            await page.get_by_label('Обновлений весов',exact=True).fill('1234')
            await page.get_by_role('button',name='Подготовить план · не запускать',exact=True).click()
            await page.get_by_text('План сохранён · обучение не запущено',exact=True).wait_for(timeout=30000)
            async with page.expect_download() as event:
                await page.get_by_role('button',name='Скачать план',exact=True).click()
            download=await event.value;await download.save_as(root/'draft.json')
            draft=json.loads((root/'draft.json').read_text())
            assert draft['steps']==1234 and draft['preparation_only'] and not draft['training_started']
            assert set(draft['split_counts'].values())=={1}
            await page.reload()
            await page.get_by_label('Открыть подготовленный план',exact=True).click();await page.get_by_role('option').first.click()
            await page.get_by_text('Сохранённый план проверен · обучение не запущено',exact=True).wait_for(timeout=30000)
            assert not await page.get_by_role('checkbox',name='Запустить обучение сейчас по показанному неизменяемому плану',exact=True).is_checked()
            assert jobs()==before and not errors and not external
            await page.screenshot(path=str(root/'prepared.png'),full_page=True)
            await page.get_by_role('checkbox',name='Запустить обучение сейчас по показанному неизменяемому плану',exact=True).check()
            await page.get_by_role('button',name='Подготовить другой план',exact=True).click()
            await expect(page.get_by_role('button',name='Запустить подготовленное обучение',exact=True)).to_be_disabled()
            assert not await page.get_by_role('checkbox',name='Запустить обучение сейчас по показанному неизменяемому плану',exact=True).is_checked()
            assert jobs()==before
            (root/'result.json').write_text(json.dumps({'actual_browser':True,'prepare_and_reopen':'PASS',
                'steps':1234,'split_counts':draft['split_counts'],'no_jobs_created':True,'training_started':False,
                'errors':errors,'external':external},indent=2))
        except Exception:
            await page.screenshot(path=str(root/'failure.png'),full_page=True);raise
        finally:
            await context.tracing.stop(path=str(root/'trace.zip'));await browser.close()

asyncio.run(main())
