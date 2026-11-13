"""Finite, reviewable simulation comparisons in the existing job queue."""
import asyncio
import json
from pathlib import Path
from nicegui import ui
from . import catalog
from .paths import DATA
from .sim_pages import STYLE

LABELS={'continue':'Продолжить','hold_only':'Подождать без помощи','correction':'Подождать и исправить'}
OUTCOMES={'success':'Успех','failure':'Неуспех','censored':'Время сравнения закончилось'}


def install(css,jobs):
    @ui.page('/sim/branches')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        ident=None;displayed=None;busy=False;import_id=None;import_status=None
        def choices():
            result={'diagnostic':'Проверка механики · удержание без обученной политики'}
            for job in catalog.rows('jobs'):
                if job['profile']==profile.value and job['status']=='completed' and job['checkpoint'] and job['policy'] in {'act','smolvla','diffusion'}:
                    result[job['id']]=job['policy'].upper()+' · '+job['id'][:8]
            policy.set_options(result,value='diagnostic')
        async def launch():
            nonlocal ident,displayed,import_id,import_status
            try:
                ident=await asyncio.to_thread(jobs.compare_sim_branches,profile.value,
                    source_job=None if policy.value=='diagnostic' else policy.value,
                    seed=int(seed.value),warmup_s=float(warmup.value),delay_s=float(delay.value),
                    correction_s=float(duration.value),horizon_s=float(horizon.value))
                displayed=None;import_id=None;import_status=None;status.set_text('Сравнение в очереди · '+ident[:8]);results.clear()
                history.set_options(saved_reports(),value=None)
            except Exception as exc:status.set_text(str(exc))
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / СРАВНЕНИЕ РЕШЕНИЙ').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Что изменит помощь в одной и той же ситуации').classes('muted')
            ui.label('MuJoCo · без обучения').classes('sim-badge')
        with ui.row():
            ui.link('← Симуляция','/sim');ui.link('Общий стенд','/sim/fleet');ui.link('Зрение','/sim/vision');ui.link('Оценщик пользы помощи','/sim/gain')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                ui.label('Одно состояние · три варианта').classes('sim-title')
                ui.label('Продолжение сохраняет очередь команд политики. Две другие ветви ждут одинаковое время: затем одна удерживает положение, другая получает ограниченную помощь. После этого обе очищают очередь и возвращаются к политике.').classes('muted')
                profile=ui.select({'tank_dock_sim':'Tank · подъезд к цели','so101_pick_place_sim':'SO-101 · кубик в коробку'},
                    value='tank_dock_sim',label='Сцена',on_change=choices).classes('w-full')
                policy=ui.select({},label='Что делает робот без помощи').classes('w-full');choices()
                ui.label('Проверка механики использует удержание вместо обученной политики. Она проверяет повторяемость, а не качество обучения. Старый checkpoint от другой версии физики будет отклонён.').classes('muted')
            with ui.column().classes('sim-panel'):
                ui.label('Бюджет сравнения').classes('sim-title')
                with ui.grid(columns=2).classes('w-full'):
                    seed=ui.number('Расстановка',value=14,min=0,step=1,precision=0).classes('w-full')
                    warmup=ui.number('До сохранения, с',value=.5,min=0,max=20,step=.5).classes('w-full')
                    delay=ui.number('Ожидание помощи, с',value=.5,min=0,max=10,step=.5).classes('w-full')
                    duration=ui.number('Исправление, с',value=5,min=.05,max=10,step=.5).classes('w-full')
                    horizon=ui.number('После сохранения, с',value=15,min=.1,max=30,step=1).classes('w-full')
                ui.label('Каждая ветвь повторяется дважды из сохранённого файла. Это секунды симуляции; время человека по ним не оценивается. Помощь оказывает геометрический алгоритм, который видит состояние симулятора.').classes('muted')
                with ui.row():
                    ui.button('Сравнить три варианта',icon='fork_right',on_click=launch)
                    ui.button('Отменить',on_click=lambda:jobs.cancel(ident) if ident else None).props('outline')
                status=ui.label('Готово к сравнению').classes('sim-status')
        def saved_reports():
            return {j['id']:j['profile']+' · '+j['id'][:8]+' · '+j['status'] for j in catalog.rows('jobs')
                    if json.loads(j['config']).get('kind')=='sim_branches'}
        def show_saved():
            nonlocal ident,displayed,import_id,import_status
            if history.value:
                ident=history.value;displayed=None;import_id=None;import_status=None;results.clear()
        history=ui.select(saved_reports(),label='Открыть сохранённое сравнение',on_change=show_saved).classes('w-full')
        results=ui.column().classes('w-full')
        def render(report):
            nonlocal import_status
            with results:
                with ui.column().classes('sim-panel'):
                    ui.label('Восстановление состояния подтверждено' if report['replay_verified'] else 'Восстановление не подтверждено').classes('sim-title')
                    ui.label('Одна независимая ситуация. Повторы не увеличивают число независимых наблюдений и не дают вероятность успеха.').classes('muted')
                    rows=[]
                    for trial in report['outcomes']:
                        if trial['repeat']==0:
                            rows.append({'branch':LABELS[trial['branch']],'outcome':OUTCOMES[trial['outcome']],
                                'duration':round(trial['sim_seconds'],2),'help':round(trial['correction_sim_s'],2)})
                    ui.table(columns=[{'name':'branch','label':'Вариант','field':'branch'},
                        {'name':'outcome','label':'Результат','field':'outcome'},
                        {'name':'duration','label':'Время симуляции, с','field':'duration'},
                        {'name':'help','label':'Помощь, с','field':'help'}],rows=rows,row_key='branch').classes('w-full')
                    ui.button('Скачать полный отчёт',icon='download',on_click=lambda:ui.download.file(DATA/'jobs'/ident/'result.json'))
                with ui.row().classes('w-full no-wrap').style('overflow-x:auto'):
                    for branch,label in LABELS.items():
                        with ui.column().classes('sim-panel').style('min-width:280px;flex:1'):
                            ui.label(label).classes('sim-title');image=DATA/'jobs'/ident/(branch+'.png')
                            if image.exists():ui.image(image)
                if report['correction_candidates']:
                    with ui.column().classes('sim-panel'):
                        ui.label('Сохранённое исправление').classes('sim-title')
                        ui.label('Только команды геометрического эксперта, без ожидания и действий политики. Все ветви этой ситуации останутся одной группой данных.').classes('muted')
                        try:
                            from .branch_corrections import candidate
                            root,meta=candidate(report)
                            paths=sorted(root.glob('*.png'))
                            with ui.row().classes('w-full'):
                                for path in (paths[0],paths[len(paths)//2],paths[-1]):ui.image(path).style('width:30%')
                            reviewed=ui.checkbox('Проверил кадры и результат · сохранить как исправление в симуляции')
                            async def accept():
                                nonlocal import_id
                                try:
                                    import_id=await asyncio.to_thread(jobs.accept_sim_correction,ident,reviewed.value)
                                    import_status.set_text('Сохранение и проверка набора в очереди')
                                except Exception as exc:import_status.set_text(str(exc))
                            ui.button('Добавить исправление в датасеты',icon='save',on_click=accept)
                            import_status=ui.label('Обучение не запускается; источник останется scripted expert.').classes('muted')
                        except ValueError as exc:ui.label(str(exc)).classes('muted')
        async def poll():
            nonlocal busy,displayed
            if ident is None or busy:return
            busy=True
            try:
                if import_id and import_status:
                    entry=jobs.get(import_id)
                    if entry['status']=='completed':
                        imported=json.loads((DATA/'jobs'/import_id/'result.json').read_text())
                        import_status.set_text('Исправление сохранено · проверка чтения '+imported['readback']+' · '+imported['episode_id'][:8])
                    elif entry['status']=='failed':import_status.set_text(jobs.log(import_id)[-1200:])
                job=jobs.get(ident)
                status.set_text({'queued':'Ожидает свободных ресурсов','running':'Сравнение трёх вариантов','completed':'Сравнение завершено',
                    'failed':'Не удалось завершить сравнение','cancelled':'Сравнение отменено'}.get(job['status'],job['status']))
                if job['status']=='completed' and displayed!=ident:
                    report=json.loads((DATA/'jobs'/ident/'result.json').read_text());results.clear();render(report);displayed=ident
                elif job['status']=='failed':status.set_text(jobs.log(ident)[-1800:])
            finally:busy=False
        ui.timer(1.,poll)
