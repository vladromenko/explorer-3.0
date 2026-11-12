"""Explorer 3.0 — one NiceGUI/FastAPI server, local assets, inert startup."""
import asyncio
import json
import os
import time
from fastapi import Request
from fastapi.responses import StreamingResponse
from pathlib import Path
from nicegui import app,ui
from nicegui.server import Server
from . import catalog
from .control import Control
from .jobs import Jobs
from .session import Session
from .preview import LivePreview
from . import handoff, tank_handoff, tank
from .network import LocalNetworkOnly
from . import viewer, local_work, local_pages, research, trials as trial_plans
from .paths import DATA, ROOT, prepare
from .profiles import PROFILES
from .resources import owners

prepare()
app.add_middleware(LocalNetworkOnly)
catalog.initialize()
control=Control()
jobs=Jobs(control)
session=Session(control)
research.initialize()
local_pages.install(jobs)
live_preview=LivePreview(session,stop_requested=lambda:Server.instance.should_exit)
ROBOT_NAMES={'so101':'SO-101','tank':'Tank','rosmaster':'ROSMASTER'}
DEVICE_PATHS={'leader':'/dev/so101-leader','follower':'/dev/so101-follower',
              'wrist':'/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0'}

def inventory():
    held=owners(list(DEVICE_PATHS.values()))
    return {k:{'present':Path(v).exists(),'owners':[o for o in held if str(Path(v).resolve()) in o['devices']]}
            for k,v in DEVICE_PATHS.items()}

def current_stage():
    if (DATA/'validation/stage-b-pass.json').exists():
        return 'C'
    return 'B' if control.authorized() else 'A'

@app.get('/api/health')
def health():
    return {'product':'Explorer 3.0','stage':current_stage(),'armed':False if control.owner is None else True,
            'local_only':local_work.enabled(),'selected_robot':control.robot,'mode':control.mode,'devices':inventory(),'data_root':str(DATA),
            'session':{'phase':session.phase,'frames':session.frame_count,'error':session.error,
                       'recording':session.record_requested,'thread_active':bool(session.thread and session.thread.is_alive()),
                       'stop_reason':control.last_stop_reason,'heartbeat_count':control.heartbeat_count,
                       'heartbeat_age_s':time.monotonic()-control.last_heartbeat if control.last_heartbeat else None},
            'compute_job':jobs.active_id,
            'preview':{'transport':'mjpeg_http','clients':live_preview.clients,'encoded_frames':live_preview.encoded}}

@app.get('/api/live/camera.mjpg')
@app.get('/api/live/wrist.mjpg')
async def wrist_preview(request: Request):
    return StreamingResponse(live_preview.stream(request),media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.get('/api/status')
def status():
    return {'health':health(),'profiles':PROFILES,'jobs':catalog.rows('jobs'),'episodes':catalog.rows('episodes')}

CSS='''
body { background:#10161f; color:#e6edf4; font-family:system-ui,sans-serif; }
.nicegui-content { max-width:1440px; margin:auto; padding:30px; width:100%; }
.hero { display:flex; align-items:center; justify-content:space-between; width:100%; gap:20px; }
.kicker { color:#7dd3c7; letter-spacing:.14em; text-transform:uppercase; font-size:12px; }
h1 { font-size:34px; margin:4px 0; font-weight:700; letter-spacing:-1px; }
.muted { color:#9aacbf; }
.fleet { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; width:100%; margin:22px 0; }
.robot { background:#192331; border:1px solid #344253; border-radius:14px; padding:20px; box-shadow:none; }
.preview { background:#111923; border:1px dashed #405167; height:125px; display:flex; align-items:center; justify-content:center; width:100%; color:#90a2b5; border-radius:8px; }
.workspace { background:#192331; border:1px solid #344253; border-radius:14px; width:100%; padding:22px; }
.selected { font-size:22px; font-weight:650; color:#9de7dd; }
.q-tab-panels { background:transparent; width:100%; }
.q-tab-panel { padding:24px 0; }
.q-table__container { background:#121d29; color:#e6edf4; width:100%; }
.q-field__native,.q-field__label { color:#cdd9e6; }
.q-tabs { width:100%; border-bottom:1px solid #344253; }
.q-btn { text-transform:none; border-radius:8px; }
.notice { padding:14px 18px; background:#2b2b20; border:1px solid #6b6236; border-radius:9px; color:#e6dca5; width:100%; }
.q-footer { background:#192331; padding:10px 30px; }
@media(max-width:800px) { .fleet {grid-template-columns:1fr;} .hero {align-items:flex-start;} .nicegui-content {padding:16px;} }
'''

def notify_error(exc):
    catalog.event('error',str(exc))
    ui.notify(str(exc),type='negative')

@ui.page('/')
def index():
    owner=ui.context.client.id
    client=ui.context.client
    ui.on('operator_heartbeat',lambda:control.heartbeat(owner))
    def lost(reason):
        if control.owner==owner:
            session.finish('aborted',reason='Браузер: '+str(reason))
    ui.on('operator_lost',lambda event:lost(event.args))
    ui.on('operator_blur',lambda:lost('окно потеряло фокус') if control.robot=='tank' else None)
    def drive_key(event):
        value=event.args
        if session.phase=='following' and control.robot=='tank':
            control.base_key(owner,value['key'],bool(value['pressed']))
    ui.on('tank_key',drive_key)
    ui.on('operator_stop',lambda:lost('клавиша STOP'))
    client.on_disconnect(lambda:lost('соединение закрыто'))
    ui.add_body_html("<script>setInterval(()=>{if(!document.hidden) emitEvent('operator_heartbeat')},250);document.addEventListener('visibilitychange',()=>{if(document.hidden) emitEvent('operator_lost','вкладка скрыта')});window.addEventListener('pagehide',()=>emitEvent('operator_lost','страница закрыта'));</script>")
    ui.add_body_html('''<script>
    window.addEventListener('blur',()=>emitEvent('operator_blur'));
    const tankKeys={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',KeyW:'up',KeyS:'down',KeyA:'left',KeyD:'right'};
    function tankKey(event,pressed){
      const pad=document.getElementById('tank-drive-pad');
      if(!pad || !pad.offsetParent) return;
      if(event.code==='Space'){event.preventDefault();emitEvent('operator_stop');return}
      if(!pressed || !['INPUT','TEXTAREA'].includes(event.target.tagName)){
        const key=tankKeys[event.code];
        if(key){event.preventDefault();if(!event.repeat)emitEvent('tank_key',{key,pressed})}
      }
    }
    window.addEventListener('keydown',e=>tankKey(e,true));
    window.addEventListener('keyup',e=>tankKey(e,false));
    </script>''')
    ui.add_css(CSS)
    ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
    ui.dark_mode().enable()
    with ui.element('div').classes('hero'):
        with ui.column().style('gap:2px'):
            ui.label('HOMEBRAIN / ЛОКАЛЬНАЯ ЛАБОРАТОРИЯ').classes('kicker')
            ui.html('<h1>Explorer 3.0</h1>')
            ui.label('Показы → обучение → проверяемые испытания').classes('muted')
        ui.button('STOP',icon='stop_circle',color='negative',on_click=lambda:control.stop('STOP из интерфейса')).props('size=lg')
    stage_label=ui.label('Этап A · программная подготовка. Движение не разрешено. Запись и физические испытания ещё не проверены.').classes('notice')
    ready=ui.label('').classes('muted')
    fleet_labels={}
    with ui.element('div').classes('fleet'):
        for ident,name in ROBOT_NAMES.items():
            with ui.card().classes('robot'):
                ui.label(name).style('font-size:22px;font-weight:650')
                label='USB обнаружены · требуется приёмка' if ident=='so101' and all(v['present'] for v in inventory().values()) else 'Подключение ещё не проверено'
                fleet_labels[ident]=ui.badge(label,color='grey-8')
                with ui.element('div').classes('preview'):
                    ui.label('Камера не открыта' if ident=='so101' else 'Подключение на этапе '+('C' if ident=='tank' else 'D'))
                ui.button('Выбрать '+name,on_click=lambda robot=ident:select(robot)).props('outline')
    with ui.element('div').classes('workspace'):
        selected=ui.label('Выбран: '+ROBOT_NAMES[control.robot]).classes('selected')
        mode_label=ui.label('Наблюдение · приводы не включены').classes('muted')
        task=ui.select({k:k for k,v in PROFILES.items() if v['robot']==control.robot},value=next(k for k,v in PROFILES.items() if v['robot']==control.robot),label='Задача').style('min-width:320px')
        with ui.column().style('width:100%') as tank_controls:
            tank_status=ui.label('Tank: связь проверяется, приводы закрыты').classes('muted')
            ui.label('Удерживайте стрелку или W/A/S/D. Отпускание — остановка; пробел — STOP.')
            ui.html('''<div id="tank-drive-pad" style="display:flex;gap:12px;touch-action:none;user-select:none">'''+''.join(
                f'''<button class="q-btn bg-primary text-white" style="min-width:64px;min-height:48px;touch-action:none"
                onpointerdown="this.setPointerCapture(event.pointerId);emitEvent('tank_key',{{key:'{key}',pressed:true}})"
                onpointerup="emitEvent('tank_key',{{key:'{key}',pressed:false}})"
                onpointercancel="emitEvent('tank_key',{{key:'{key}',pressed:false}})"
                onlostpointercapture="emitEvent('tank_key',{{key:'{key}',pressed:false}})">{label}</button>'''
                for key,label in [('left','←'),('up','↑'),('down','↓'),('right','→')])+'</div>')
        with ui.tabs() as tabs:
            management=ui.tab('Управление',icon='tune')
            record=ui.tab('Запись',icon='fiber_manual_record')
            datasets=ui.tab('Датасеты',icon='video_library')
            training=ui.tab('Обучение',icon='model_training')
            trials=ui.tab('Испытания',icon='science')
        with ui.tab_panels(tabs,value=management):
            with ui.tab_panel(management):
                ui.label('Режим владения устройством').style('font-size:19px')
                with ui.row():
                    ui.button('Наблюдение',on_click=lambda:control.stop('Наблюдение'))
                    async def transfer(restore=False):
                        try:
                            local_work.require_hardware_mode()
                            with control.lock:
                                if control.maintenance or (session.thread and session.thread.is_alive()):
                                    raise RuntimeError('Сначала STOP и дождитесь закрытия сессии')
                                control.maintenance=True
                            try:
                                if control.robot=='tank':
                                    operation=tank_handoff.restore if restore else tank_handoff.enter
                                    await asyncio.to_thread(operation,control)
                                elif control.robot=='so101':
                                    if restore:await asyncio.to_thread(handoff.restore,control)
                                    else:await asyncio.to_thread(handoff.enter,control,list(DEVICE_PATHS.values()))
                                else:raise RuntimeError('Драйвер ещё не подключён')
                            finally:control.maintenance=False
                            ui.notify('Прежние службы восстановлены' if restore else 'Ресурсы освобождены; движение не началось')
                        except Exception as exc:notify_error(exc)
                    async def take_control():await transfer()
                    async def old_control():await transfer(restore=True)
                    take_button=ui.button('Explorer 3.0',on_click=take_control)
                    def teleop_check():
                        try:session.start(owner,task.value,'teleop-check',record=False)
                        except Exception as exc:notify_error(exc)
                    teleop_button=ui.button('Проверить телеперацию',on_click=teleop_check)
                    old_button=ui.button('Прежнее управление',on_click=old_control)
                ui.label('Проверка телеперации не записывает показ. После проверки нажмите STOP.').classes('muted')
                device_table=ui.table(columns=[{'name':'name','label':'Устройство','field':'name'}, {'name':'state','label':'Фактическое состояние','field':'state'}],
                    rows=[],row_key='name')
                ui.label('SO-101 управляется Leader. Tank и ROSMASTER подключаются по очереди.').classes('muted')
            with ui.tab_panel(record):
                task_title=ui.label('Кубик → коробка → отвести руку').style('font-size:20px')
                task_description=ui.label('').classes('muted')
                collection=ui.input('Сессия сбора (одна для серии показов)',value=time.strftime('%Y-%m-%d')+'-session-01')
                scene=ui.input('Расстановка',value='scene-01')
                category=ui.select({'demonstration':'Исходный показ','correction':'Отдельный сегмент коррекции'},value='demonstration',label='Тип записи')
                ui.label('Коррекцию начинайте отдельной записью после остановки предыдущего исполнения: границы источников не смешиваются.').classes('muted')
                def begin():
                    try:
                        session.start(owner,task.value,scene.value,collection_id=collection.value,category=category.value)
                        ui.notify('Запуск разрешённой сессии')
                    except Exception as exc:
                        notify_error(exc)
                with ui.row():
                    start_button=ui.button('Начать показ',on_click=begin)
                    ui.button('Закончить · успех',on_click=lambda:session.finish('success'))
                    ui.button('Закончить · неуспех',on_click=lambda:session.finish('failure'))
                    ui.button('Прервать',on_click=lambda:session.finish('aborted'))
                recording_status=ui.label('Нет активной записи')
                camera_view=ui.html('<img src="/api/live/camera.mjpg" alt="Камера выбранного робота: поток появится при запуске" style="width:100%;max-width:640px;aspect-ratio:4/3;object-fit:contain;background:#10161f">')
                ui.label('Перед первым показом: передача управления и короткая проверка телеперации.').classes('notice')
                chart=ui.echart({'backgroundColor':'transparent','textStyle':{'color':'#9aacbf'},'tooltip':{'trigger':'axis'},
                    'legend':{'data':['Команда','Измерение'],'textStyle':{'color':'#9aacbf'}},
                    'xAxis':{'type':'value','name':'секунды'},'yAxis':{'type':'value'},
                    'series':[{'name':'Команда','type':'line','data':[]},{'name':'Измерение','type':'line','data':[]}]}).style('height:240px;width:100%')
            with ui.tab_panel(datasets):
                ui.label('Исходные эпизоды сохраняются. Исключение влияет только на подбор обучения.').classes('muted')
                episode_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('id','Эпизод'),('profile','Задача'),('outcome','Исход'),('frames','Кадры'),('category','Тип'),('session','Сессия'),('scene','Расстановка'),('excluded','Исключён')]],rows=[],row_key='id',selection='single')
                def exclusion():
                    for row in episode_table.selected:
                        catalog.exclude(row['id'],not bool(row['excluded']))
                    refresh()
                ui.button('Исключить / вернуть в обучение',on_click=exclusion)
                def view_selected():
                    if episode_table.selected:
                        ui.navigate.to('/episodes/'+episode_table.selected[0]['id'])
                    else:
                        ui.notify('Выберите эпизод в таблице')
                ui.button('Посмотреть',on_click=view_selected)
                async def validate_selected():
                    from .validation import validate
                    try:
                        if not episode_table.selected:raise ValueError('Выберите эпизод')
                        result=await asyncio.to_thread(validate,episode_table.selected[0]['id'])
                        ui.notify(f"Запись прочитана: {result['frames']} кадров, {result['actual_hz']:.1f} Гц")
                    except Exception as exc:notify_error(exc)
                ui.button('Проверить данные и видео',on_click=validate_selected)
                collection_info=ui.label().classes('muted')
            with ui.tab_panel(training):
                ui.label('Один trainer · отдельная политика для каждой задачи').style('font-size:19px')
                ui.label('ACT: upstream LeRobot 0.6.1. Smoke: 50 шагов; это проверка тракта, не навык.').classes('muted')
                async def enqueue_policy(policy,smoke):
                    try:
                        ident=await asyncio.to_thread(jobs.enqueue,task.value,policy,smoke)
                        ui.notify('Задание '+ident[:8]+' поставлено в очередь; состав данных зафиксирован')
                    except Exception as exc:notify_error(exc)
                async def act_smoke():await enqueue_policy('act',True)
                async def act_full():await enqueue_policy('act',False)
                async def smol_smoke():await enqueue_policy('smolvla',True)
                async def smol_full():await enqueue_policy('smolvla',False)
                with ui.row():
                    ui.button('ACT · smoke 50 шагов',on_click=act_smoke)
                    ui.button('ACT · основное обучение',on_click=act_full)
                    smol_button=ui.button('SmolVLA · smoke',on_click=smol_smoke)
                    smol_full_button=ui.button('SmolVLA · основное обучение',on_click=smol_full)
                ui.label('Основной рецепт: 100 000 шагов; checkpoint каждые 5 000. Validation отдельно; test не используется для подбора. Незавершённое обучение продолжается только явно.').classes('muted')
                job_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('id','Задание'),('profile','Задача'),('policy','Модель'),('status','Состояние'),('checkpoint','Checkpoint'),('error','Ошибка')]],rows=[],row_key='id',selection='single')
                def selected_job_action(action):
                    try:
                        if not job_table.selected:raise ValueError('Выберите задание')
                        ident=job_table.selected[0]['id']
                        if action=='open':ui.navigate.to('/jobs/'+ident)
                        elif action=='resume':jobs.resume(ident);ui.notify('Продолжение в очереди')
                        elif action=='cancel':jobs.cancel(ident)
                        elif action=='benchmark':
                            result=jobs.benchmark(ident);ui.notify('Проверка нагрузки в очереди '+result[:8])
                        elif action in {'evaluate','test'}:
                            result=jobs.evaluate(ident,'test' if action=='test' else 'validation');ui.notify('Диагностика в очереди '+result[:8])
                    except Exception as exc:notify_error(exc)
                with ui.row():
                    ui.button('Журнал и результат',on_click=lambda:selected_job_action('open'))
                    ui.button('Продолжить с checkpoint',on_click=lambda:selected_job_action('resume'))
                    ui.button('Отменить выбранное',on_click=lambda:selected_job_action('cancel'))
                    ui.button('Проверить checkpoint на записях',on_click=lambda:selected_job_action('evaluate'))
                    ui.button('Финальный test выбранного checkpoint',on_click=lambda:selected_job_action('test'))
                    ui.button('Измерить нагрузку 3 копий политики',on_click=lambda:selected_job_action('benchmark'))
            with ui.tab_panel(trials):
                ui.label('Программная готовность ≠ checkpoint ≠ физический навык').style('font-size:19px')
                prepared_trial=[None]
                trial_details=ui.code('Выберите запись в Датасетах или обучение с checkpoint, затем подготовьте план.',language='json').style('max-height:260px;overflow:auto;width:100%')
                def saved_plans():
                    return {p.stem:json.loads(p.read_text())['profile']+' · '+json.loads(p.read_text())['kind']+' · '+p.stem[:8] for p in sorted((DATA/'trials').glob('*.json'))}
                def choose_plan(event):
                    if event.value:
                        prepared_trial[0]=event.value
                        trial_details.set_content(json.dumps(trial_plans.load(event.value),ensure_ascii=False,indent=2))
                trial_select=ui.select(saved_plans(),label='Сохранённый план испытания',on_change=choose_plan).style('min-width:400px')
                async def prepare_trial(kind):
                    try:
                        selected_rows=episode_table.selected if kind=='replay' else job_table.selected
                        if not selected_rows:raise ValueError('Сначала выберите запись или задание обучения в соответствующей вкладке')
                        plan=await asyncio.to_thread(trial_plans.prepare,kind,selected_rows[0]['id'])
                        prepared_trial[0]=plan['id']
                        trial_select.set_options(saved_plans(),value=plan['id'])
                        trial_details.set_content(json.dumps(plan,ensure_ascii=False,indent=2))
                        ui.notify('План сохранён; движения не запускались')
                    except Exception as exc:notify_error(exc)
                async def prepare_replay():await prepare_trial('replay')
                async def prepare_policy():await prepare_trial('policy')
                def start_trial():
                    try:
                        if not prepared_trial[0]:raise ValueError('Сначала подготовьте план')
                        session.start_trial(owner,prepared_trial[0],scene.value)
                    except Exception as exc:notify_error(exc)
                with ui.row():
                    ui.button('Подготовить повтор записи',on_click=prepare_replay)
                    ui.button('Подготовить испытание checkpoint',on_click=prepare_policy)
                    trial_start_button=ui.button('Запустить разрешённое испытание',on_click=start_trial)
                with ui.row():
                    ui.button('Посмотреть запись',on_click=view_selected)

                ui.label('Физический replay и rollout недоступны в локальном режиме. Checkpoint можно проверить на записях во вкладке Обучение.').classes('muted')
                ui.link('Исследование вмешательств: пилот, журнал, оценщики','/research')
                profile_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('profile','Задача'),('record','Запись'),('train','Обучение'),('skill','Физический навык')]],
                    rows=[{'profile':k,
                           'record':'проверена' if v['recording_status']=='verified' else 'не проверено',
                           'train':'smoke проверен · 50 шагов' if v['training_status']=='smoke_verified' else 'не проверено',
                           'skill':'не проверено'} for k,v in PROFILES.items()])
        with ui.expansion('Журнал событий и ошибок',icon='receipt_long').style('width:100%'):
            event_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('level','Уровень'),('message','Сообщение')]],rows=[],row_key='id')
    with ui.footer():
        ui.label('Homebrain · автономное локальное приложение · питание роботов: состояние неизвестно')
    def select(robot):
        if session.thread and session.thread.is_alive():
            control.stop('Смена робота: сначала закрытие текущей сессии')
            ui.notify('STOP выполняется. Выберите робота после остановки.');return
        try:control.select(robot)
        except Exception as exc:
            notify_error(exc);return
        selected.set_text('Выбран: '+ROBOT_NAMES[robot])
        with session.lock:session.latest=None
        camera_view.set_content(f'<img src="/api/live/camera.mjpg?robot={robot}" alt="Камера выбранного робота появится при запуске" style="width:100%;max-width:640px;object-fit:contain;background:#10161f">')
        for series in chart.options['series']:series['data']=[]
        chart.update();last_frame[0]=None
        options={k:k for k,v in PROFILES.items() if v['robot']==robot}
        task.set_options(options,value=next(iter(options)))
    last_frame=[None]
    def refresh():
        if PROFILES[task.value]['robot']!=control.robot:
            options={k:k for k,v in PROFILES.items() if v['robot']==control.robot}
            task.set_options(options,value=next(iter(options)))
        authorized=control.authorized()
        trial_start_button.set_enabled(authorized and not local_work.enabled() and bool(prepared_trial[0]))
        supported=control.robot in {'so101','tank'}
        idle=not control.maintenance and not (session.thread and session.thread.is_alive())
        start_button.set_enabled(authorized and supported and idle)
        take_button.set_enabled(authorized and supported and idle)
        teleop_button.set_enabled(authorized and supported and idle)
        old_button.set_enabled(authorized and supported and idle)
        tank_controls.set_visibility(control.robot=='tank')
        task_title.set_text('Подъехать к видимой цели → остановиться в зоне' if control.robot=='tank' else 'Кубик → коробка → отвести руку')
        task_description.set_text({'so101':'Wrist RGB и измеренные суставы; цель — отправленная команда Follower.',
            'tank':'Передняя RGB; состояние — предыдущая команда, измеренной скорости колёс нет.',
            'rosmaster':'Каналы взяты из CubeBot 8090. Камера и контракт обратной связи Explorer требуют проверки на роботе.'}[control.robot])
        selected.set_text('Выбран: '+ROBOT_NAMES[control.robot])
        smol_pass=(DATA/'validation/smolvla-pass.json').exists()
        smol_button.set_enabled(smol_pass and task.value=='so101_pick_place')
        smol_full_button.set_enabled(smol_pass and task.value=='so101_pick_place')
        collection_info.set_text(local_pages.collection_status(task.value))
        for ident,badge in fleet_labels.items():
            badge.set_text('Не опрашивается · локальный режим' if local_work.enabled() else ('Выбран' if ident==control.robot else 'Не выбран'))
        act_pass=(DATA/'validation/act-pass.json').exists()
        ready.set_text('Программный smoke: ACT '+('PASS' if act_pass else 'не проверен')+' · SmolVLA '+('PASS' if smol_pass else 'не проверен')+' · физические навыки не проверены')
        if local_work.enabled():
            stage_label.set_text('Локальный режим · роботы не опрашиваются. Датасеты, обучение и анализ доступны без включения устройств.')
        elif control.robot=='tank' and (DATA/'validation/tank-motion-incident.json').exists():
            stage_label.set_text('Tank: движение заблокировано после отказа STOP. Диагностика; физическая приёмка не пройдена.')
        elif current_stage()=='C':
            stage_label.set_text('Этап C · подключение Tank. Приёмка SO-101 завершена; физический навык не проверен.')
        elif authorized:
            stage_label.set_text('Этап B · разрешённый физический блок. Движение начинается только явной кнопкой.')
        phases={'idle':'Остановлено','preparing':'Подготовка — пока не двигайте Leader','aligning':'Плавное совмещение — держите Leader неподвижно','following':'Следование — можно двигать Leader','error':'Ошибка','saving':'Сохранение эпизода — дождитесь завершения','trial':'Испытание — действует STOP, исход отмечается оператором'}
        if control.robot=='tank':
            phases.update(preparing='Подготовка камеры и питания — пока не нажимайте стрелки',aligning='Подготовка',following='Управление готово — можно удерживать стрелки')
        if control.robot=='tank' and session.phase=='idle':
            phases['idle']='Командная сессия закрыта; физическая остановка не измеряется'
        phase_text='Передача управления — дождитесь завершения' if control.maintenance else phases.get(session.phase,session.phase)
        recording_status.set_text(session.error or (phase_text+' · кадров: '+str(session.frame_count)+((' · '+control.last_stop_reason) if control.last_stop_reason else '')))
        device_table.rows=[{'name':k,'state':('занято' if v['owners'] else 'USB обнаружено, не открыто') if v['present'] else 'отсутствует'} for k,v in inventory().items()]
        device_table.update()
        profile_table.rows=local_pages.profile_statuses();profile_table.update()
        episode_table.rows=catalog.rows('episodes');episode_table.update()
        job_table.rows=catalog.rows('jobs');job_table.update()
        event_table.rows=catalog.rows('events')[:20];event_table.update()
        mode_label.set_text(control.mode+' · '+phase_text)
    def refresh_preview():
        with session.lock:
            sample=session.latest
        if sample and sample['frame_id']!=last_frame[0]:
            state_label='Предыдущая команда (оценка)' if control.robot=='tank' else 'Измерение'
            chart.options['series'][1]['name']=state_label
            chart.options['legend']['data']=['Команда',state_label]
            for series,key in zip(chart.options['series'],('action','observation.state')):
                series['data'].append([sample['frame_id']/PROFILES[task.value]['fps'],sample[key][0]])
                series['data']=series['data'][-300:]
            chart.update()
            last_frame[0]=sample['frame_id']
    async def refresh_tank():
        if local_work.enabled():
            tank_status.set_text('Tank выключен по сообщению оператора; сетевой опрос отключён')
        elif control.robot=='tank':
            def probe():
                with tank.client() as http:
                    response=http.get('/health');response.raise_for_status();return response.json()
            try:
                observed=await asyncio.to_thread(probe)
                voltage=observed['battery_voltage']
                fresh=observed['battery_received_ns'] is not None and observed['monotonic_ns']-observed['battery_received_ns']<2_000_000_000
                battery=f'{voltage:.2f} В' if fresh else 'неизвестно (нет свежего измерения)'
                tank_status.set_text('Tank по Wi-Fi · '+('порт управления открыт' if observed['hardware_open'] else 'порт управления закрыт; остановка моторов не измеряется')+' · питание: '+battery)
            except Exception as exc:tank_status.set_text('Tank недоступен: '+str(exc))
    refresh()
    ui.timer(2,refresh_tank)
    ui.timer(2,refresh)
    ui.timer(.25,refresh_preview)

app.on_startup(lambda:catalog.event('info','Explorer 3.0 запущен без открытия аппаратуры'))
app.on_shutdown(lambda:control.stop('Служба останавливается'))
app.on_shutdown(jobs.cancel)
app.timer(.1,control.watchdog)
app.timer(2,jobs.tick)
ui.run(host=os.environ.get('EXPLORER3_HOST','127.0.0.1'),port=int(os.environ.get('EXPLORER3_PORT','8030')),title='Explorer 3.0',
       reload=False,show=False,tailwind=False,language='ru',favicon='🧭')
