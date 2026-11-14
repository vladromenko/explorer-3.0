"""Actual browser preparation, no model calls, simulation or robot requests."""
import asyncio,json,os,re,sqlite3
from pathlib import Path
from urllib.parse import urlsplit
from playwright.async_api import async_playwright,expect

DATA=Path(os.environ['EXPLORER3_CHECK_DATA']);BASE=os.environ['EXPLORER3_CHECK_URL'];OUT=Path(os.environ['EXPLORER3_CHECK_OUTPUT'])
if 'validation' not in DATA.parts:raise ValueError('Isolated validation storage required')


def jobs():
    with sqlite3.connect(DATA/'catalog.sqlite3') as db:return db.execute('SELECT id,status,config FROM jobs ORDER BY id').fetchall()


async def main():
    before=jobs();errors=[];external=[];OUT.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context(viewport={'width':1360,'height':1000})
        await context.tracing.start(screenshots=True,snapshots=True)
        async def local(route):
            if urlsplit(route.request.url).hostname not in {None,'localhost','127.0.0.1'}:
                external.append(route.request.url);await route.abort()
            else:await route.continue_()
        await context.route('**/*',local)
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto(BASE+'/team/observations')
            await expect(page.get_by_role('heading',name='Наблюдения команды')).to_be_visible()
            await page.get_by_role('button',name='Открыть кадр',exact=True).click()
            await expect(page.get_by_text('Кадр проверен и сохранён',exact=False)).to_be_visible(timeout=15000)
            await expect(page.get_by_role('button',name='Найти объект · YOLOE',exact=True)).to_be_enabled()
            await page.screenshot(path=str(OUT/'observations.png'),full_page=True)
            await page.get_by_label('Запись или показ',exact=True).click()
            await page.get_by_role('option',name='so101_pick_place · показ',exact=False).first.click()
            await page.get_by_role('button',name='Открыть кадр',exact=True).click()
            await expect(page.get_by_text('SO101 · so101/wrist',exact=True)).to_be_visible(timeout=15000)
            await page.goto(BASE+'/team/comparison')
            await page.get_by_role('button',name='Подготовить сравнение · не запускать',exact=True).click()
            await expect(page.get_by_text('План сохранён. Обучение и симуляция не запущены.',exact=True)).to_be_visible(timeout=15000)
            await expect(page.get_by_role('button',name='Выполнить один опыт',exact=True)).to_be_disabled()
            await page.get_by_role('button',name='Показать отчёт',exact=True).click()
            await expect(page.get_by_text('M3: 0/10',exact=False)).to_be_visible()
            await page.set_viewport_size({'width':390,'height':844})
            await page.screenshot(path=str(OUT/'comparison-mobile.png'),full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
            await page.goto(BASE+'/team/real')
            await page.get_by_role('button',name='Подготовить последовательность · не запускать',exact=True).click()
            await expect(page.get_by_text('Выберите от одного до четырёх разных испытаний',exact=True)).to_be_visible()
            for name in ('SO-101 · Кубик в коробку','ROSMASTER · Подъезд к цели'):
                await page.get_by_label('Испытания в порядке выполнения',exact=True).click()
                await page.get_by_role('option',name=re.compile('^'+name+' · Повтор записи')).first.click()
                await page.keyboard.press('Escape')
            for robot in ('so101','rosmaster'):
                await page.get_by_label('Исходные наблюдения каждого робота',exact=True).click()
                await page.get_by_role('option',name=re.compile('^'+robot.upper()+' ·')).first.click()
                await page.keyboard.press('Escape')
            await page.get_by_role('button',name='Подготовить последовательность · не запускать',exact=True).click()
            await expect(page.get_by_text('План сохранён. Реальные движения ещё не разрешены.',exact=True)).to_be_visible(timeout=15000)
            await expect(page.get_by_role('button',name='Разрешить эту миссию и начать',exact=True)).to_be_disabled()
            await page.screenshot(path=str(OUT/'real-mission-prepared-mobile.png'),full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
            await page.get_by_label('Задача',exact=True).fill('Изменённая задача')
            await expect(page.get_by_text('Выбор изменился. Подготовьте новый план перед запуском.',exact=True)).to_be_visible()
            await page.goto(BASE+'/sim/recovery-data')
            await expect(page.get_by_role('heading',name='Восстановительные показы')).to_be_visible()
            await page.get_by_role('button',name='Подготовить план · не запускать',exact=True).click()
            await expect(page.get_by_text('Нужен завершённый ACT и измеренная ошибка на validation',exact=True)).to_be_visible()
            await expect(page.get_by_role('button',name='Собрать две серии в симуляции',exact=True)).to_be_disabled()
            assert before==jobs() and not errors and not external
            result={'passed':True,'source':'copies of original recordings','frames_opened':['rosmaster','so101'],
                'frozen_comparison_prepared':True,'incomplete_report_visible':True,'jobs_unchanged':True,
                'real_graph_from_original_trials':True,'changed_plan_invalidated':True,'recovery_missing_baseline_rejected':True,
                'model_calls':0,'simulation_runs':0,'physical_commands':0,'errors':errors,'external':external}
            (OUT/'cognitive-preparation-result.json').write_text(json.dumps(result,indent=2))
        except Exception:
            await page.screenshot(path=str(OUT/'cognitive-preparation-failure.png'),full_page=True)
            raise
        finally:
            await context.tracing.stop(path=str(OUT/'cognitive-preparation-trace.zip'));await browser.close()


asyncio.run(main())
