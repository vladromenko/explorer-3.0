"""Finite joint policy runs, calibrated monitor selection and retained reports."""
import asyncio
import base64
import json
from pathlib import Path
from nicegui import ui
from . import catalog,fleet_runs
from .paths import DATA
from .sim_pages import STYLE

NAMES={'so101':'SO-101','tank':'Tank','rosmaster':'ROSMASTER · модель'}


def install(css,jobs):
    @ui.page('/sim/fleet/policies')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        ident=None;busy=False;displayed=None;imports={};selectors={};enabled={};supports={};last_image=None
        available=fleet_runs.selections()
        def support_choices(name):
            choices={'none':'Оценщик не подключён'}
            if selectors[name].value=='fixture':choices={'none':'Тестовый оценщик · программный пример'}
            else:
                for job in catalog.rows('jobs'):
                    cfg=json.loads(job['config'])
                    if job['status']=='completed' and cfg.get('kind')=='support_calibration' and cfg.get('source_job')==selectors[name].value:
                        choices[job['id']]='Калибровка по отложенным данным · '+job['id'][:8]
            supports[name].set_options(choices,value='none')
        async def launch():
            nonlocal ident
            try:
                selected={n:selectors[n].value for n in selectors if enabled[n].value}
                support={n:supports[n].value for n in selected if supports[n].value!='none'}
                ident=await asyncio.to_thread(jobs.fleet_execute,selected,horizon_s=float(horizon.value),seed=int(seed.value),
                    method=method.value,delay_s=float(delay.value),help_s=float(budget.value),software_probe=probe.value,support=support,
                    gain_job=gain.value if gain.value!='none' else None)
                status.set_text('Проверка в очереди · '+ident[:8]);links.clear()
                with links:ui.link('Журнал задания и отмена','/jobs/'+ident)
                saved.set_options(history(),value=None)
            except Exception as exc:status.set_text(str(exc))
        async def cancel():
            if ident:await asyncio.to_thread(jobs.cancel,ident)
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / СОВМЕСТНОЕ ИСПОЛНЕНИЕ').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Отдельные политики · одна физическая сцена · один слот помощи').classes('muted')
            ui.button('STOP',icon='stop_circle',color='negative',on_click=cancel).props('size=lg')
        with ui.row():
            ui.link('← Общий стенд','/sim/fleet');ui.link('Подготовка обучения','/sim/training');ui.link('Сравнение вариантов помощи','/sim/branches');ui.link('Оценщик пользы','/sim/gain')
        with ui.element('div').classes('sim-panel'):
            ui.label('Что исполняет каждый робот').classes('sim-title')
            ui.label('Физические роботы не подключаются. Каждая политика получает только свою камеру и состояние. Во время расчёта действий симуляционные часы ждут; скорость реального управления этим не подтверждается.').classes('muted')
            for name,label in NAMES.items():
                with ui.column().classes('w-full'):
                    enabled[name]=ui.checkbox(label,value=name!='rosmaster')
                    choices={'fixture':'Программный пример · без обученных весов'}
                    for job in available:
                        if job['profile']==fleet_runs.PROFILES[name]:choices[job['id']]=job['policy'].upper()+' · '+job['id'][:12]+(' · короткая старая проверка' if json.loads(job['config']).get('smoke') else '')
                    selectors[name]=ui.select(choices,value='fixture',label='Политика',on_change=lambda name=name:support_choices(name)).classes('w-full')
                    supports[name]=ui.select({'none':'Тестовый оценщик · программный пример'},value='none',label='Оценщик запроса помощи').classes('w-full')
            probe=ui.checkbox('Разрешить программную проверку старых весов на новой физике · без оценки навыка')
            gain_choices={'none':'Оценщик пользы не подключён'}
            for job in catalog.rows('jobs'):
                if job['status']=='completed' and json.loads(job['config']).get('kind')=='branch_gain_train':
                    gain_choices[job['id']]='Прогноз в симуляции · '+job['id'][:8]
            gain=ui.select(gain_choices,value='none',label='Польза помощи · только наблюдение').classes('w-full')
            ui.label('Для рабочих весов калибровка привязана к конкретной модели и независимым данным. Если её нет, оценщик не подменяется выдуманными порогами. В программном примере пороги тестовые и явно отмечены.').classes('muted')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                ui.label('Общая сцена').classes('sim-title');preview=ui.image().props('no-spinner')
                status=ui.label('Готово к конечной проверке').classes('sim-status');links=ui.column()
            with ui.column().classes('sim-panel'):
                ui.label('Бюджет исполнения и помощи').classes('sim-title')
                horizon=ui.number('Время симуляции, с',value=1,min=.2,max=20,step=.1)
                seed=ui.number('Расстановка',value=1,min=0,precision=0)
                delay=ui.number('Задержка помощи, с',value=.1,min=0,max=10,step=.1)
                budget=ui.number('Максимум помощи за запрос, с',value=1,min=.05,max=10,step=.1)
                method=ui.select({'fleet_thrifty':'Fleet-ThriftyDAgger · риск и новизна','random':'Fleet-DAgger · случайный приоритет'},value='fleet_thrifty',label='Кому помочь первым')
                ui.button('Исполнить политики · без обучения',icon='play_arrow',on_click=launch)
                ui.label('Помощь здесь оказывает заданный геометрический алгоритм. Это программная проверка очереди, а не эксперимент с человеком. Запрос ждёт не более трёх секунд симуляции.').classes('muted')
        def history():
            return {j['id']:j['id'][:8]+' · '+j['status'] for j in catalog.rows('jobs') if json.loads(j['config']).get('kind')=='fleet_execution'}
        def open_saved():
            nonlocal ident,last_image
            if saved.value:ident=saved.value;last_image=None
        saved=ui.select(history(),label='Открыть сохранённый прогон',on_change=open_saved).classes('w-full')
        table=ui.table(columns=[{'name':'robot','label':'Робот','field':'robot'},{'name':'source','label':'Политика','field':'source'},
            {'name':'monitor','label':'Оценщик','field':'monitor'},{'name':'request','label':'Запрос помощи','field':'request'},
            {'name':'priority','label':'Соответствие / риск','field':'priority'},
            {'name':'gain','label':'Прогноз пользы · sim','field':'gain'},
            {'name':'actions','label':'Команд политики','field':'actions'}],rows=[],row_key='robot').classes('w-full')
        detail=ui.label('').classes('sim-status')
        with ui.row():
            ui.button('Скачать отчёт',icon='download',on_click=lambda:ui.download.file(DATA/'jobs'/ident/'result.json') if ident and (DATA/'jobs'/ident/'result.json').exists() else None).props('outline')
            ui.button('Скачать команды',icon='download',on_click=lambda:ui.download.file(DATA/'jobs'/ident/'commands.json') if ident and (DATA/'jobs'/ident/'commands.json').exists() else None).props('outline')
        corrections=ui.column().classes('sim-panel');corrections.set_visibility(False)
        ui.label('Оценщик соответствия показам и прогноз пользы — разные величины. Прогноз малого оценщика появится только при подключении обученной модели; он не выбирает робота и не подтверждает пользу человека.').classes('muted')
        async def poll():
            nonlocal busy,last_image,displayed
            if ident is None or busy:return
            busy=True
            try:
                job=jobs.get(ident);root=DATA/'jobs'/ident
                path=root/('result.json' if job['status']=='completed' else 'progress.json')
                state=json.loads(path.read_text()) if path.exists() else {}
                phases={'loading':'Загрузка политик','running':'Совместное исполнение','checking_restore':'Проверка восстановления','completed':'Прогон завершён'}
                text=phases.get(state.get('phase') if job['status'] not in {'failed','cancelled'} else None,{'completed':'Прогон завершён','queued':'Ожидает свободных ресурсов','cancelled':'Остановлено','failed':'Ошибка исполнения'}.get(job['status'],job['status']))
                status.set_text(text+(' · '+str(round(state['time_s'],3))+' с' if 'time_s' in state else ''))
                if job['status']=='failed':detail.set_text(jobs.log(ident)[-1600:])
                elif state.get('policies'):
                    rows=[]
                    for name,policy in state['policies'].items():
                        monitor=state.get('monitors',{}).get(name)
                        rows.append({'robot':NAMES[name],'source':'Программный пример' if policy['kind']=='numerical_fixture' else policy['policy_type'].upper(),
                            'gain':round(state['gain_scores'][name]['gain'],3) if 'gain' in state.get('gain_scores',{}).get(name,{}) else 'нет оценки',
                            'monitor':('Просит помощь' if monitor['mode']=='operator' else 'Политика принята') if monitor else 'Не подключён',
                            'priority':f"{monitor['support']:.3f} / {monitor['risk'] if monitor['risk'] is not None else 'вне области'}" if monitor else 'нет оценки',
                            'request':{'pending':'Ожидает','assisting':'Помощь','returned':'Возврат к политике','expired':'Окно помощи истекло','budget_exhausted':'Бюджет исчерпан'}.get(state['requests'].get(name,{}).get('status'),'нет'),'actions':state['counts'][name]['policy']})
                    table.rows=rows;table.update()
                    verified=state.get('replay',{}).get('verified')
                    detail.set_text('Помощь: '+NAMES.get(state.get('operator'),'никому')+(' · Полное восстановление проверено' if verified else ''))
                if job['status']=='completed' and displayed!=ident:
                    displayed=ident;corrections.clear();corrections.set_visibility(True)
                    with corrections:
                        ui.label('Коррекции для следующего раунда').classes('sim-title')
                        segments=state.get('correction_segments',[])
                        if not segments:ui.label('В этом прогоне экспертных исправлений не было.').classes('muted')
                        for index,segment in enumerate(segments):
                            ui.label(NAMES[segment['agent']]+' · '+str(segment['frames'])+' кадров · '+segment['outcome'])
                            try:
                                from .fleet_corrections import validate
                                segment_root=Path(segment['spool']);validate(segment_root)
                                frames=sorted(segment_root.glob('*.png'))
                                with ui.row():
                                    for path in (frames[0],frames[len(frames)//2],frames[-1]):ui.image(path).style('width:30%')
                                reviewed=ui.checkbox('Проверил исправление и успешный исход задачи')
                                message=ui.label('Сохранение не запускает обучение.').classes('muted')
                                async def accept(index=index,reviewed=reviewed,message=message,source_job=ident):
                                    try:
                                        created=await asyncio.to_thread(jobs.accept_fleet_correction,source_job,index,reviewed.value)
                                        imports[created]=message;message.set_text('Сохранение в очереди')
                                    except Exception as exc:message.set_text(str(exc))
                                ui.button('Сохранить отдельное исправление',icon='save',on_click=accept).props('outline')
                            except ValueError as exc:ui.label(str(exc)+' · сегмент сохранён только для аудита.').classes('muted')
                        ui.link('Подготовить следующий раунд обучения вручную →','/sim/training')
                for imported,label in list(imports.items()):
                    entry=jobs.get(imported)
                    if entry['status']=='completed':
                        report=json.loads((DATA/'jobs'/imported/'result.json').read_text())
                        label.set_text('Сохранено · чтение '+report['readback']+' · '+report['episode_id'][:8]);imports.pop(imported)
                    elif entry['status']=='failed':label.set_text(jobs.log(imported)[-800:]);imports.pop(imported)
                frame=root/'overview.jpg'
                if frame.exists() and frame.stat().st_mtime_ns!=last_image:
                    last_image=frame.stat().st_mtime_ns;preview.set_source('data:image/jpeg;base64,'+base64.b64encode(frame.read_bytes()).decode())
            finally:busy=False
        ui.timer(.5,poll)
