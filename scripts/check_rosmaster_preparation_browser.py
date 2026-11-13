"""Live Homebrain offline page: read local catalogue, never train or contact robots."""
import asyncio,json,sqlite3
from pathlib import Path
from playwright.async_api import async_playwright
DATA=Path('/home/jetsonnx/.local/share/explorer3')
OUT=DATA/'validation/corrective-b/offline-browser'
def jobs():
    with sqlite3.connect(DATA/'catalog.sqlite3') as db:return db.execute('select id from jobs order by id').fetchall()
async def main():
    OUT.mkdir(exist_ok=True);before=jobs();errors=[];external=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=await browser.new_context(viewport={'width':1440,'height':1100})
        await ctx.tracing.start(screenshots=True,snapshots=True)
        async def route(req):
            from urllib.parse import urlsplit
            if urlsplit(req.request.url).hostname not in {None,'localhost','127.0.0.1'}:external.append(req.request.url);await req.abort()
            else:await req.continue_()
        await ctx.route('**/*',route);page=await ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto('http://127.0.0.1:8030/rosmaster/preparation')
            await page.get_by_text('Пригодных показов: 0',exact=False).wait_for(timeout=60000)
            await page.get_by_label('Исходные веса из симуляции',exact=True).click()
            await page.get_by_role('option').first.click()
            await page.get_by_role('button',name='Подготовить два плана · без запуска',exact=True).click()
            await page.get_by_text('Выбранный показ отсутствует или не допущен',exact=True).wait_for(timeout=60000)
            await page.get_by_role('button',name='Подготовить сверку отклика',exact=True).click()
            await page.get_by_text('Выберите пригодные записи измеренного движения',exact=True).wait_for()
            await page.get_by_role('button',name='Запустить дообучение',exact=True).click()
            await page.get_by_text('Нужны готовый план и отдельное подтверждение запуска',exact=True).wait_for()
            await page.get_by_label('Задача',exact=True).click()
            await page.get_by_role('option',name='Шасси · подъезд к цели',exact=True).click()
            await page.get_by_text('Пригодных показов: 0',exact=False).wait_for(timeout=60000)
            await page.screenshot(path=str(OUT/'page.png'),full_page=True)
            await page.goto('http://127.0.0.1:8030/rosmaster/control')
            await page.get_by_text('локальная подготовка',exact=True).wait_for(timeout=15000)
            response=await page.request.get('http://127.0.0.1:8030/api/rosmaster/camera.mjpg')
            assert response.status==503
            await page.screenshot(path=str(OUT/'offline-control.png'),full_page=True)
            assert before==jobs() and not errors and not external
            (OUT/'result.json').write_text(json.dumps({'actual_browser':True,'source_checkpoint_selected':True,
                'missing_real_data_explained':True,'no_training_without_plan_and_confirmation':True,
                'both_profiles_checked':True,'jobs_unchanged':True,'robot_access':False,'offline_camera_http':503,
                'offline_control_status':'PASS','errors':errors,'external':external},indent=2))
        finally:
            await ctx.tracing.stop(path=str(OUT/'trace.zip'));await browser.close()
asyncio.run(main())
