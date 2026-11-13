"""No-data and no-consent workflow on actual Chromium; never launches training."""
import asyncio,json,sqlite3
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path('/home/jetsonnx/.local/share/explorer3/validation')
def jobs():
    with sqlite3.connect(ROOT/'browser-sim/catalog.sqlite3') as db:
        return db.execute('select id from jobs order by id').fetchall()
async def main():
    before=jobs();output=ROOT/'corrective-a/browser-gain';output.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1100});errors=[];external=[]
        async def route(request):
            from urllib.parse import urlsplit
            if urlsplit(request.request.url).hostname not in {None,'localhost','127.0.0.1'}:external.append(request.request.url);await request.abort()
            else:await request.continue_()
        await context.route('**/*',route);page=await context.new_page();page.on('pageerror',lambda x:errors.append(str(x)))
        await page.goto('http://127.0.0.1:8035/sim/gain')
        await page.get_by_text('Подходящих сравнений пока нет.',exact=False).wait_for()
        await page.get_by_role('button',name='Подготовить план · без обучения',exact=True).click()
        await page.get_by_text('Нет завершённых сопоставимых сравнений',exact=True).wait_for()
        await page.get_by_role('button',name='Запустить обучение оценщика',exact=True).click()
        await page.get_by_text('Нужны готовый план и явное подтверждение запуска',exact=True).wait_for()
        await page.screenshot(path=str(output/'no-data.png'),full_page=True)
        assert before==jobs() and not errors and not external
        (output/'result.json').write_text(json.dumps({'actual_browser':True,'no_data_explanation':'PASS',
            'no_training_without_plan_and_consent':'PASS','jobs_unchanged':True,'errors':errors,'external':external},indent=2))
        await browser.close()
asyncio.run(main())
