"""Offline ROSMASTER workbench: local data, transfer and response plans."""
import asyncio,json
from nicegui import ui
from . import rosmaster_preparation as prep,transfer_preparation as transfer,training_drafts,catalog
from .paths import DATA
from .sim_pages import STYLE

LABELS={'rosmaster_pick_place':'Рука · кубик в коробку','rosmaster_dock':'Шасси · подъезд к цели'}

def install(css,jobs):
    @ui.page('/rosmaster/preparation')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        comparison=None;identification=None;busy=False
        async def refresh():
            nonlocal busy,comparison,identification
            if busy:return
            busy=True;status.set_text('Проверяю локальные записи и версии весов…')
            try:
                comparison=None;identification=None;consent.value=False;summary.clear();model_links.clear()
                history.set_options(saved(),value=None)
                identification_plans.set_options(saved_identifications(),value=None)
                audit=await asyncio.to_thread(prep.report);part=audit['profiles'][target.value]
                shows.set_options({e['id']:e['id'][:8]+' · '+str(e['frames'])+' кадров' for e in part['episodes'] if e['eligible']},value=[])
                sources.set_options({s['id']:s['policy'].upper()+' · '+s['id'][:8] for s in part['sources'] if s['eligible']},value=None)
                motion.set_options({e['id']:e['id'][:8] for e in part['motion_episodes']},value=[])
                table.rows=[{'id':e['id'][:8],'reason':e['reason']} for e in part['episodes']];table.update()
                status.set_text(f"Пригодных показов: {part['eligible_shows']} · независимых групп: {part['independent_groups']}. Для пилота нужно минимум 3 группы. Для настройки отклика: {part['motion_groups']} из 2 групп.")
                if not part['episodes']:table.rows=[{'id':'—','reason':'Настоящих показов пока нет; физические проверки отложены'}];table.update()
                identified.set_options(identification_history(),value=None)
                response_table.rows=[];response_table.update()
                response_note.set_text('Выберите готовую проверку отклика этой задачи.')
                blocked.set_text(' · '.join(s['id'][:8]+': '+s['reason'] for s in part['sources'] if not s['eligible']))
            except Exception as exc:status.set_text(str(exc))
            finally:busy=False
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / ROSMASTER').classes('kicker');ui.html('<h1>Подготовка без робота</h1>')
                ui.label('Записи · перенос из симуляции · проверка отклика').classes('muted')
            ui.label('Без подключения и обучения').classes('sim-badge')
        with ui.row():
            ui.link('Explorer','/');ui.link('Симуляция','/sim');ui.link('Панель робота','/rosmaster/control')
        with ui.column().classes('sim-panel'):
            target=ui.select(LABELS,value='rosmaster_pick_place',label='Задача',on_change=refresh).classes('w-full')
            ui.button('Проверить локальные данные',icon='refresh',on_click=refresh)
            status=ui.label('Читаю только файлы Homebrain').classes('sim-status')
            table=ui.table(columns=[{'name':'id','label':'Запись','field':'id'},{'name':'reason','label':'Пригодность','field':'reason'}],rows=[],row_key='id').classes('w-full')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                ui.label('Из симуляции — на реальные показы').classes('sim-title')
                ui.label('Два одинаковых бюджета и один набор реальных данных: дообучение из симуляции и сравнение с обучением с нуля. Симуляционные кадры не выдаются за физические. Совпадение каналов не подтверждает точность модели или навык.').classes('muted')
                sources=ui.select({},label='Исходные веса из симуляции').classes('w-full')
                blocked=ui.label('').classes('muted')
                shows=ui.select({},multiple=True,value=[],label='Настоящие успешные показы').classes('w-full')
                with ui.row():
                    steps=ui.number('Обновлений весов',value=2000,min=50,max=1000000,precision=0)
                    batch=ui.number('Размер пакета',value=1,min=1,max=32,precision=0)
                seed=ui.number('Seed',value=1000,min=0,precision=0)
                device=ui.select({'cpu':'CPU','cuda':'GPU Homebrain'},value='cuda',label='Вычисления')
                async def prepare_transfer():
                    nonlocal comparison
                    try:
                        if not sources.value:raise ValueError('Сначала нужны совместимые веса и реальные показы; обучение сейчас не запускается')
                        ident,comparison=await asyncio.to_thread(transfer.prepare,jobs.get(sources.value),target.value,shows.value,
                            steps=int(steps.value),batch=int(batch.value),seed=int(seed.value),device=device.value)
                        present(comparison);history.set_options(saved(),value=ident);status.set_text('Два плана сохранены · обучение не запускалось')
                    except Exception as exc:status.set_text(str(exc))
                ui.button('Подготовить два плана · без запуска',icon='description',on_click=prepare_transfer)
            with ui.column().classes('sim-panel'):
                ui.label('Сверить движение модели с записями').classes('sim-title')
                ui.label('Настройка по одним сессиям и расстановкам, проверка по другим. Диагностические клипы и записи без измеренного движения не подходят. Результат не применяется к модели автоматически.').classes('muted')
                motion=ui.select({},multiple=True,value=[],label='Записи движения для проверки модели').classes('w-full')
                async def prepare_identification():
                    nonlocal identification
                    try:
                        identification,_=await asyncio.to_thread(prep.prepare_identification,target.value,motion.value)
                        identification_plans.set_options(saved_identifications(),value=identification)
                        status.set_text('План сверки отклика сохранён · расчёт не запущен')
                        show_identification_plan()
                    except Exception as exc:status.set_text(str(exc))
                ui.button('Подготовить сверку отклика',on_click=prepare_identification).props('outline')
                def saved_identifications():
                    return {p.stem:p.stem[:10] for p in sorted((DATA/'identification-plans').glob('*.json'))
                        if json.loads(p.read_text()).get('profile')==target.value}
                def show_identification_plan():
                    model_links.clear()
                    if identification:
                        plan=prep.load_identification(identification)
                        with model_links:
                            ui.label('Сохранено записей: '+str(len(plan['entries']))+' · '+LABELS[plan['profile']])
                            ui.button('Скачать план сверки',on_click=lambda ident=identification:ui.download.file(DATA/'identification-plans'/(ident+'.json')))
                def open_identification():
                    nonlocal identification
                    try:
                        identification=identification_plans.value
                        show_identification_plan()
                    except Exception as exc:
                        identification=None;status.set_text(str(exc))
                identification_plans=ui.select(saved_identifications(),label='Открыть план сверки',on_change=open_identification).classes('w-full')
                model_links=ui.column()
                async def identify():
                    try:
                        if identification is None:raise ValueError('Сначала подготовьте план по реальным записям')
                        ident=await asyncio.to_thread(jobs.identify_rosmaster,identification)
                        with model_links:ui.link('Ход расчёта и результат','/jobs/'+ident)
                    except Exception as exc:status.set_text(str(exc))
                ui.button('Рассчитать отклик по плану',on_click=identify).props('outline')
                ui.label('Это расчёт параметров отклика по файлам, без движения и без обучения моторной политики.').classes('muted')
        def identification_history():
            return {j['id']:LABELS[j['profile']]+' · '+j['id'][:8] for j in catalog.rows('jobs')
                if j['status']=='completed' and j['profile']==target.value and json.loads(j['config']).get('kind')=='rosmaster_identification'}
        def show_identification():
            try:
                if not identified.value:return
                result=json.loads((DATA/'jobs'/identified.value/'result.json').read_text());rows=[]
                for name,c in result['channels'].items():
                    error=c.get('holdout_rmse',{})
                    rows.append({'channel':name,'nominal':error.get('nominal_sim'),'model':error.get('simulator_form'),
                        'full':error.get('identified'),'units':c.get('units','канал не двигался')})
                response_table.rows=rows;response_table.update()
                response_note.set_text('Проверка на отдельных записях: '+str(len(result['holdout_episodes']))+
                    '. Меньшая ошибка лучше; параметры автоматически не применены. Это не обучение политики.')
            except Exception as exc:response_note.set_text(str(exc))
        with ui.column().classes('sim-panel'):
            identified=ui.select(identification_history(),label='Готовая проверка отклика',on_change=show_identification).classes('w-full')
            response_note=ui.label('Здесь появится сравнение исходной и уточнённой модели на отдельных реальных записях.').classes('muted')
            response_table=ui.table(columns=[{'name':k,'field':k,'label':label} for k,label in
                [('channel','Канал'),('nominal','Исходная ошибка'),('model','Уточнение модели'),('full','Отклик с задержкой'),('units','Единицы')]],rows=[],row_key='channel').classes('w-full')
            ui.button('Скачать сравнение отклика',on_click=lambda:ui.download.file(DATA/'jobs'/identified.value/'result.json') if identified.value else None).props('outline')
        def saved():return {p.stem:p.stem[:10] for p in sorted((DATA/'transfer-plans').glob('*.json'))
            if json.loads(p.read_text()).get('target')==target.value}
        def present(value):
            summary.clear();consent.value=False
            with summary:
                for key,label in [('transfer_draft','Дообучение из симуляции'),('scratch_draft','Сравнение с нуля')]:
                    draft=training_drafts.load(value[key])
                    ui.label(label+' · '+str(draft['frames']['train'])+' обучающих кадров · '+str(draft['steps'])+' обновлений')
                    ui.button('Скачать: '+label,on_click=lambda ident=value[key]:ui.download.file(DATA/'training-drafts'/(ident+'.json'))).props('outline')
        def open_saved():
            nonlocal comparison
            try:
                if history.value:
                    from .transfer_preparation import load_comparison
                    comparison=load_comparison(history.value);present(comparison)
            except Exception as exc:status.set_text(str(exc))
        history=ui.select(saved(),label='Открыть сохранённое сравнение',on_change=open_saved).classes('w-full')
        summary=ui.column().classes('sim-panel')
        consent=ui.checkbox('Я отдельно запускаю выбранное обучение сейчас')
        async def start(key):
            try:
                if not consent.value or comparison is None:raise ValueError('Нужны готовый план и отдельное подтверждение запуска')
                ident=await asyncio.to_thread(jobs.start_prepared_training,comparison[key]);consent.value=False
                with summary:ui.link('Ход обучения и отмена','/jobs/'+ident)
            except Exception as exc:status.set_text(str(exc))
        with ui.row():
            ui.button('Запустить дообучение',on_click=lambda:start('transfer_draft')).props('outline')
            ui.button('Запустить сравнение с нуля',on_click=lambda:start('scratch_draft')).props('outline')
        ui.timer(.1,refresh,once=True)
