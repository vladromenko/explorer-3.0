"""Prepare common intervention-estimator data; training is a separate explicit action."""
import asyncio
import json
from nicegui import ui
from . import catalog,gain_preparation
from .paths import DATA
from .sim_pages import STYLE


def install(css,jobs):
    @ui.page('/sim/gain')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        plan_id=None
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / ПОЛЬЗА ПОМОЩИ').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Общий оценщик · отдельные моторные политики').classes('muted')
            ui.label('Подготовка без обучения').classes('sim-badge')
        with ui.row():
            ui.link('Сравнение вариантов','/sim/branches');ui.link('Совместное исполнение','/sim/fleet/policies')
        with ui.column().classes('sim-panel'):
            ui.label('Какие ситуации использовать').classes('sim-title')
            ui.label('Оценщик сравнивает шанс завершения с ограниченной программной помощью и без неё. Это оценка в симуляции, а не измеренный эффект человека. Повторы и одна расстановка при разных задержках остаются вместе; исходы ещё не завершённых ветвей не подменяются неуспехом.').classes('muted')
            available={};rejected=[]
            for job in catalog.rows('jobs'):
                if job['status']!='completed' or json.loads(job['config']).get('kind')!='sim_branches':continue
                path=DATA/'jobs'/job['id']/'result.json'
                try:
                    gain_preparation.candidate(json.loads(path.read_text()))
                    available[str(path)]=job['profile']+' · '+job['id'][:8]
                except (ValueError,OSError,KeyError) as exc:rejected.append({'job':job['id'][:8],'reason':str(exc)})
            selected=ui.select(available,multiple=True,value=[],label='Завершённые сравнения').classes('w-full')
            if not available:ui.label('Подходящих сравнений пока нет. Диагностическое удержание и старые отчёты без истории решения сюда не добавляются.').classes('muted')
            if rejected:
                with ui.expansion('Почему другие сравнения не подходят').classes('w-full'):
                    ui.table(columns=[{'name':'job','label':'Сравнение','field':'job'},
                        {'name':'reason','label':'Причина','field':'reason'}],rows=rejected,row_key='job')
            physical=ui.checkbox('Учитывать доступные измерения исполнения',value=True)
            ui.label('У Tank без подтверждённых энкодеров измерения движения отсутствуют: оценщик получает маску отсутствия, а не скорость из скрытой физики.').classes('muted')
            async def prepare():
                nonlocal plan_id
                try:
                    plan_id,plan=await asyncio.to_thread(gain_preparation.prepare,selected.value,include_physical=physical.value)
                    show(plan);saved.set_options(history(),value=plan_id)
                except Exception as exc:status.set_text(str(exc))
            ui.button('Подготовить план · без обучения',icon='description',on_click=prepare)
            status=ui.label('Минимум пять независимых сцен; успехи и неуспехи обеих ветвей в обучающей части.').classes('sim-status')
        def history():
            return {p.stem:p.stem[:12] for p in sorted((DATA/'gain-plans').glob('*.json'))}
        def show(plan):
            status.set_text('План подготовлен · обучение не запущено')
            rows=[{'role':role,'scenes':len({r['scene_group'] for r in plan['rows'] if r['split']==role}),
                'rows':sum(r['split']==role for r in plan['rows'])} for role in ('train','validation','test')]
            table.rows=rows;table.update()
        def open_plan():
            nonlocal plan_id
            if saved.value:
                try:plan_id=saved.value;show(gain_preparation.load(plan_id))
                except Exception as exc:status.set_text(str(exc))
        saved=ui.select(history(),label='Сохранённый план оценщика',on_change=open_plan).classes('w-full')
        table=ui.table(columns=[{'name':'role','label':'Часть','field':'role'},
            {'name':'scenes','label':'Независимых сцен','field':'scenes'},
            {'name':'rows','label':'Ситуаций','field':'rows'}],rows=[],row_key='role').classes('w-full')
        ui.button('Скачать план',icon='download',on_click=lambda:ui.download.file(DATA/'gain-plans'/(plan_id+'.json')) if plan_id else None).props('outline')
        with ui.column().classes('sim-panel'):
            ui.label('Обучение — отдельный запуск').classes('sim-title')
            consent=ui.checkbox('Сейчас запускаю обучение малого оценщика на выбранном плане')
            async def train():
                try:
                    if not consent.value or not plan_id:raise ValueError('Нужны готовый план и явное подтверждение запуска')
                    ident=await asyncio.to_thread(jobs.train_prepared_gain,plan_id)
                    status.set_text('Обучение запущено оператором · '+ident[:8])
                    with links:ui.link('Результат и отмена','/jobs/'+ident)
                except Exception as exc:status.set_text(str(exc))
            ui.button('Запустить обучение оценщика',on_click=train).props('outline')
            ui.label('Две малые модели, фиксированный предел 1000 итераций, нормировка только по train. Готовый результат можно отдельно подключить к общему исполнению для наблюдения за прогнозами. Очередью помощи он автоматически не управляет.').classes('muted')
            links=ui.column()
