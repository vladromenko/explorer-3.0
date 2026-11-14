"""Read copied recordings, exercise preparation errors; never press training start."""
import asyncio,json,os,sqlite3
from pathlib import Path
from playwright.async_api import async_playwright,expect

DATA=Path(os.environ['EXPLORER3_CHECK_DATA']).resolve()
BASE=os.environ['EXPLORER3_CHECK_URL']
OUT=Path(os.environ['EXPLORER3_CHECK_OUTPUT'])
if 'validation' not in DATA.parts:raise ValueError('Use isolated validation storage')


def jobs():
    with sqlite3.connect(DATA/'catalog.sqlite3') as db:
        return db.execute('select id,status,config from jobs order by id').fetchall()


async def main():
    OUT.mkdir(parents=True,exist_ok=True);before=jobs();errors=[];external=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1440,'height':1100})
        await context.tracing.start(screenshots=True,snapshots=True)
        async def local(route):
            from urllib.parse import urlsplit
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:
                external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local)
        page=await context.new_page();page.on('pageerror',lambda x:errors.append(str(x)))
        try:
            await page.goto(BASE+'/training/prepare')
            await expect(page.get_by_text('Реальные записи',exact=True)).to_be_visible()
            start=page.get_by_role('button',name='Запустить подготовленное обучение',exact=True)
            await expect(start).to_be_disabled()
            await page.get_by_label('Показы и исправления для нового плана',exact=True).click()
            await expect(page.get_by_role('option')).to_have_count(3)
            for option in await page.get_by_role('option').all():await option.click()
            await page.keyboard.press('Escape')
            await page.get_by_role('button',name='Подготовить план · не запускать',exact=True).click()
            await expect(page.get_by_text('Нужно минимум 50',exact=False)).to_be_visible()
            await expect(start).to_be_disabled()
            await page.screenshot(path=str(OUT/'real-preparation-desktop.png'),full_page=True)
            await page.get_by_label('Задача',exact=True).click()
            await page.get_by_role('option',name='rosmaster_dock',exact=True).click()
            await page.get_by_label('Модель',exact=True).click()
            await expect(page.get_by_role('option')).to_have_count(1)
            await expect(page.get_by_role('option')).to_have_text('ACT · базовая политика')
            await page.keyboard.press('Escape')
            await page.set_viewport_size({'width':390,'height':844})
            await page.screenshot(path=str(OUT/'real-preparation-mobile.png'),full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
            await page.goto(BASE+'/datasets/guide')
            await expect(page.get_by_role('heading',name='Explorer 3.0: собрать датасет и подготовить обучение',exact=True)).to_be_visible()
            await expect(page.get_by_text('50 пригодных успешных показов',exact=False)).to_be_visible()
            await page.screenshot(path=str(OUT/'dataset-guide-mobile.png'),full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
            assert jobs()==before and not errors and not external
            (OUT/'real-preparation-result.json').write_text(json.dumps({'passed':True,
                'actual_browser':True,'real_source_copies':True,'insufficient_data_rejected':True,
                'supported_recipes_only':True,'mobile_overflow':False,'guide_rendered':True,
                'jobs_unchanged':True,'training_started':False,'physical_commands':0,
                'full_real_50_episode_plan':'not_verified_missing_corpus','errors':errors,'external':external},indent=2))
        except Exception:
            await page.screenshot(path=str(OUT/'real-preparation-failure.png'),full_page=True);raise
        finally:
            await context.tracing.stop(path=str(OUT/'real-preparation-trace.zip'));await browser.close()


asyncio.run(main())
