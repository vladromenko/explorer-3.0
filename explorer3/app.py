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
from . import viewer, local_work, local_pages, research, rosmaster_monitor, rosmaster_control, rosmaster_console, leader8090, trials as trial_plans
from agents.tank.motion_permission import bench_authorized
from .paths import DATA, ROOT, prepare
from .profiles import PROFILES
from .resources import owners
from .recording import code_version,source_snapshot

prepare()
PROCESS_BUILD={'pid':os.getpid(),'started_at':time.time(),'git_head':code_version(),
               'startup_disk_snapshot':source_snapshot()}
(DATA/'logs'/f'startup-{os.getpid()}.json').write_text(json.dumps(PROCESS_BUILD,indent=2))
app.add_middleware(LocalNetworkOnly)
catalog.initialize()
control=Control()
jobs=Jobs(control)
session=Session(control)
research.initialize()
local_pages.install(jobs)
from . import sensor_pages
sensor_pages.install()
live_preview=LivePreview(session,stop_requested=lambda:Server.instance.should_exit)
ROBOT_NAMES={'so101':'SO-101','tank':'Tank','rosmaster':'ROSMASTER'}
DEVICE_PATHS={'leader':'/dev/so101-leader','follower':'/dev/so101-follower',
              'wrist':'/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0'}

def inventory():
    held=owners(list(DEVICE_PATHS.values()))
    return {k:{'present':Path(v).exists(),'owners':[o for o in held if str(Path(v).resolve()) in o['devices']]}
            for k,v in DEVICE_PATHS.items()}

def current_stage():
    if local_work.workflow_stage():return local_work.workflow_stage()
    if local_work.homebrain_only():return 'A'
    if (DATA/'validation/stage-b-pass.json').exists():
        return 'C'
    return 'B' if control.authorized() else 'A'

@app.get('/api/health')
def health():
    return {'product':'Explorer 3.0','stage':current_stage(),'armed':control.owner is not None or rosmaster_control.operator.lease is not None,
            'rosmaster_manual_enabled':rosmaster_control.enabled(),'rosmaster_manual_active':rosmaster_control.operator.lease is not None,
            'local_only':local_work.enabled(),'selected_robot':control.robot,'mode':control.mode,'devices':inventory(),'data_root':str(DATA),
            'session':{'phase':session.phase,'frames':session.frame_count,'error':session.error,
                       'recording':session.record_requested,'thread_active':bool(session.thread and session.thread.is_alive()),
                       'stop_reason':control.last_stop_reason,'heartbeat_count':control.heartbeat_count,
                       'heartbeat_age_s':time.monotonic()-control.last_heartbeat if control.last_heartbeat else None},
            'compute_job':jobs.active_id,'process_build':{k:v for k,v in PROCESS_BUILD.items() if k!='startup_disk_snapshot'},
            'startup_source_sha256':PROCESS_BUILD['startup_disk_snapshot']['sha256'],
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

async def stop_all():
    for job in catalog.rows('jobs'):
        if job['status'] in {'queued','running'} and json.loads(job['config']).get('kind')=='cognitive_simulation':
            await asyncio.to_thread(jobs.cancel,job['id'])
    control.stop('STOP из интерфейса')
    if rosmaster_control.operator.lease is not None:
        await asyncio.to_thread(rosmaster_control.operator.stop,'Главный STOP Explorer')

from . import sim_pages
sim_pages.install(stop_all,CSS,jobs)
from . import fleet_pages
fleet_pages.install(stop_all,CSS)
from .cognitive import pages as cognitive_pages
cognitive_pages.install(CSS,jobs,physical_session=session)
from . import fleet_live_pages
fleet_live_pages.install(stop_all,CSS)
from . import vision_pages
vision_pages.install(CSS,jobs)
from . import branching_pages
branching_pages.install(CSS,jobs)
from . import training_pages
training_pages.install(CSS,jobs)
from . import fleet_policy_pages
fleet_policy_pages.install(CSS,jobs)
from . import gain_pages
gain_pages.install(CSS,jobs)
from . import rosmaster_preparation_pages
rosmaster_preparation_pages.install(CSS,jobs)

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
        if session.phase in {'following','paused','correction'} and control.robot=='tank':
            control.base_key(owner,value['key'],bool(value['pressed']),value.get('source'))
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
        if(key){event.preventDefault();if(!event.repeat)emitEvent('tank_key',{key,pressed,source:'key:'+event.code})}
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
        ui.button('STOP · всё управление Explorer',icon='stop_circle',color='negative',on_click=stop_all).props('size=lg')
    ui.link('Симуляция MuJoCo · Homebrain', '/sim')
    ui.link('Tank · камеры, глубина и лидар', '/tank/sensors')
    ui.link('ROSMASTER · датчики и окружение', '/rosmaster/sensors')
    ui.link('Общий обзор роботов · связь и датчики', '/fleet/live')
    ui.link('Команда · совместная задача и память', '/team')
    ui.link('Собрать датасет · инструкция', '/datasets/guide')
    ui.link('Подготовить обучение на реальных записях', '/training/prepare')
    ui.label('РЕАЛЬНЫЕ РОБОТЫ').classes('kicker')
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
                    ui.label('Физическая проверка на этапе '+{'so101':'D','tank':'C','rosmaster':'B'}[ident])
                ui.button('Выбрать '+name,on_click=lambda robot=ident:select(robot)).props('outline')
                if ident=='rosmaster':
                    ui.link('Подготовка без робота', '/rosmaster/preparation')
                    ui.link('Открыть управление', '/rosmaster/control')
                    ui.link('Камера и диагностика', '/rosmaster')
    with ui.element('div').classes('workspace'):
        selected=ui.label('Выбран: '+ROBOT_NAMES[control.robot]).classes('selected')
        mode_label=ui.label('Наблюдение · приводы не включены').classes('muted')
        task=ui.select({k:v.get('label',k) for k,v in PROFILES.items() if v['robot']==control.robot},value=next(k for k,v in PROFILES.items() if v['robot']==control.robot),label='Задача').style('min-width:320px')
        def teleop_check():
            try:session.start(owner,task.value,'teleop-check',record=False)
            except Exception as exc:notify_error(exc)
        with ui.column().style('width:100%') as tank_controls:
            tank_status=ui.label('Tank: связь проверяется, приводы закрыты').classes('muted')
            with ui.row():
                tank_start=ui.button('Включить управление',on_click=teleop_check)
                ui.button('STOP',color='negative',on_click=lambda:lost('STOP Tank'))
            with ui.column().classes('notice text-red-4') as tank_link_controls:
                tank_link_notice=ui.label('Связь с моторной платой прервалась')
                tank_stationary=ui.checkbox('Tank стоит; подтверждаю остановку гусениц')
                async def resume_tank_link():
                    try:
                        if session.thread and session.thread.is_alive():raise RuntimeError('Дождитесь сохранения прерванной записи')
                        await asyncio.to_thread(tank.acknowledge_motor_link,tank_stationary.value)
                        tank_stationary.value=False
                        ui.notify('Связь восстановлена. Нажмите «Включить управление» для новой сессии.')
                    except Exception as exc:notify_error(exc)
                ui.button('Продолжить после восстановления USB',on_click=resume_tank_link)
            tank_link_controls.set_visibility(False)
            tank_motion_state=ui.label('Управление выключено').classes('notice')
            tank_pilot_state=ui.label('').classes('notice')
            with ui.row() as correction_controls:
                def correction_done():
                    if session.pilot and control.owner==owner:session.pilot.correction_done.set()
                def emergency():
                    if session.pilot and control.owner==owner:
                        session.pilot.emergency=True
                        session.finish('aborted','Аварийное вмешательство оператора')
                ui.button('Коррекция готова',on_click=correction_done)
                ui.button('Аварийное вмешательство · STOP',color='negative',on_click=emergency)
            tank_help=ui.label('').classes('notice')
            ui.html('<img src="/api/live/camera.mjpg?robot=tank" alt="Камера появится после включения управления" style="width:100%;aspect-ratio:8/5;object-fit:contain;background:#10161f">').style('width:100%;max-width:640px')
            ui.label('Стрелки или W/A/S/D; две клавиши вместе — движение по дуге. ↙ и ↘ — назад по дуге в выбранную сторону. Отпускание — остановка; пробел — STOP.')
            ui.html('''<div id="tank-drive-pad" style="display:grid;grid-template-columns:repeat(3,72px);gap:8px;touch-action:none;user-select:none">'''+''.join(
                f'''<button class="q-btn bg-primary text-white" style="min-width:64px;min-height:48px;touch-action:none"
                aria-label="{title}"
                onpointerdown="this.setPointerCapture(event.pointerId);emitEvent('tank_key',{{key:'{key}',pressed:true,source:'pointer:'+event.pointerId}})"
                onpointerup="emitEvent('tank_key',{{key:'{key}',pressed:false,source:'pointer:'+event.pointerId}})"
                onpointercancel="emitEvent('tank_key',{{key:'{key}',pressed:false,source:'pointer:'+event.pointerId}})"
                onlostpointercapture="emitEvent('tank_key',{{key:'{key}',pressed:false,source:'pointer:'+event.pointerId}})">{label}</button>'''
                if key else '<button class="q-btn bg-negative text-white" onclick="emitEvent(\'operator_stop\')" aria-label="STOP">STOP</button>'
                for key,label,title in [('up_left','↖','Вперёд влево'),('up','↑','Вперёд'),('up_right','↗','Вперёд вправо'),
                                        ('left','←','Поворот влево'),(None,'STOP','STOP'),('right','→','Поворот вправо'),
                                        ('down_left','↙','Назад влево'),('down','↓','Назад'),('down_right','↘','Назад вправо')])+'</div>')
        with ui.tabs() as tabs:
            management=ui.tab('Управление',icon='tune')
            record=ui.tab('Запись',icon='fiber_manual_record')
            datasets=ui.tab('Датасеты',icon='video_library')
            training=ui.tab('Обучение',icon='model_training')
            trials=ui.tab('Испытания',icon='science')
        with ui.tab_panels(tabs,value=management):
            with ui.tab_panel(management):
                ownership_title=ui.label('Подключение SO-101').style('font-size:19px')
                with ui.row() as ownership_controls:
                    ui.button('Отключить управление',on_click=lambda:control.stop('Управление отключено'))
                    async def transfer(restore=False,confirmed=False):
                        try:
                            if not restore and control.robot!='so101' and not (control.robot=='tank' and bench_authorized(DATA/'configs')):
                                local_work.require_hardware_mode(control.robot)
                            with control.lock:
                                if control.maintenance or (session.thread and session.thread.is_alive()):
                                    raise RuntimeError('Сначала STOP и дождитесь закрытия сессии')
                                control.maintenance=True
                            try:
                                if control.robot=='tank':
                                    operation=tank_handoff.restore if restore else tank_handoff.enter
                                    await asyncio.to_thread(operation,control)
                                elif control.robot=='so101':
                                    if restore:
                                        def close_so_readers():
                                            if rosmaster_control.operator.mirror:
                                                raise RuntimeError('Сначала STOP зеркала ROSMASTER')
                                            rosmaster_control.operator.stop_leader()
                                            leader8090.leader8090.stop()
                                        await asyncio.to_thread(handoff.restore,control,close_devices=close_so_readers)
                                    else:await asyncio.to_thread(handoff.enter,control,list(DEVICE_PATHS.values()),confirmed=confirmed)
                                else:raise RuntimeError('Драйвер ещё не подключён')
                            finally:control.maintenance=False
                            ui.notify('Прежние службы восстановлены' if restore else 'Ресурсы освобождены; движение не началось')
                        except Exception as exc:notify_error(exc)
                    async def take_control():
                        if control.robot!='so101':
                            await transfer()
                            return
                        with ui.dialog() as confirmation, ui.card():
                            ui.label('Старая система :8020 будет остановлена до возврата')
                            ui.label('Движение не запускается. Для возврата используйте «Вернуть в :8020».')
                            ui.button('Отмена',on_click=lambda:confirmation.submit(False))
                            ui.button('Подтверждаю — взять SO-101',on_click=lambda:confirmation.submit(True))
                        if await confirmation:await transfer(confirmed=True)
                    async def old_control():await transfer(restore=True)
                    take_button=ui.button('Взять SO-101 в Explorer',on_click=take_control)
                    teleop_button=ui.button('Проверить телеперацию',on_click=teleop_check)
                    old_button=ui.button('Вернуть в :8020',on_click=old_control)
                ownership_help=ui.label('Телеперация без записи. Для записи перейдите во вкладку «Запись».').classes('muted')
                with ui.expansion('Приложение телефона и диагностика').style('width:100%') as tank_settings:
                    tank_restore=ui.button('Вернуть управление приложению Hiwonder',on_click=old_control)
                    ui.label('После STOP эта кнопка освобождает камеру и шасси для штатного приложения телефона. Следующее включение в Explorer само подключит Tank обратно.').classes('muted')
                device_table=ui.table(columns=[{'name':'name','label':'Устройство','field':'name'}, {'name':'state','label':'Фактическое состояние','field':'state'}],
                    rows=[],row_key='name')
                # ROSMASTER: the same full control page (camera, keyboard/gamepad, poses, Leader mirror)
                # embedded here; it connects the robot on «Включить управление». Loaded only while shown.
                with ui.column().style('width:100%') as rosmaster_panel:
                    ui.label('ROSMASTER: щёлкните по панели, затем клавиши/джойстик. Уход фокуса из панели = STOP.').classes('muted')
                    rosmaster_frame=ui.element('iframe').props('src="about:blank" allow="gamepad"').style(
                        'width:100%;height:1250px;border:0;border-radius:10px;background:#0e1116')
            with ui.tab_panel(record):
                task_title=ui.label('Кубик → коробка → отвести руку').style('font-size:20px')
                task_description=ui.label('').classes('muted')
                collection=ui.input('Сессия сбора (одна для серии показов)',value=time.strftime('%Y-%m-%d')+'-session-01')
                scene=ui.input('Расстановка',value='scene-01')
                ui.label('Для отдельной независимой серии нужны новые сессия и расстановка. Изменение имени без изменения условий не делает данные независимыми.').classes('muted')
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
                ui.link('Сначала подготовить и проверить план без запуска', '/training/prepare')
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
                with ui.expansion('Пилот человеческих коррекций Tank').style('width:100%') as pilot_options:
                    use_pilot=ui.checkbox('Включить пилот для выбранного испытания',value=False)
                    study_select=ui.select({},label='Пилот с проверенными задержками').style('min-width:320px')
                    ui.label('Включайте после проверки политики и безопасной точки ожидания. Случайно выбранная коррекция длится не более 10 секунд; затем снова действует политика.').classes('muted')
                    def reload_studies():
                        study_select.set_options({s['id']:s['id'][:8]+' · '+s['policy_sha256'][:8]
                            for s in research.studies() if s['profile']=='tank_dock' and s.get('timing_accepted')})
                    ui.button('Обновить список пилотов',on_click=reload_studies)
                    reload_studies()
                def start_trial():
                    try:
                        if not prepared_trial[0]:raise ValueError('Сначала подготовьте план')
                        if use_pilot.value and not study_select.value:raise ValueError('Выберите проверенный пилот')
                        session.start_trial(owner,prepared_trial[0],scene.value,study_select.value if use_pilot.value else None)
                    except Exception as exc:notify_error(exc)
                with ui.row():
                    ui.button('Подготовить повтор записи',on_click=prepare_replay)
                    ui.button('Подготовить испытание checkpoint',on_click=prepare_policy)
                    trial_start_button=ui.button('Запустить разрешённое испытание',on_click=start_trial)
                    def authorize_trial():
                        try:
                            if not prepared_trial[0]:raise ValueError('Сначала подготовьте или выберите план')
                            trial_plans.authorize(prepared_trial[0]);ui.notify('План разрешён на 3 часа. ROSMASTER: запуск в панели «Управление»')
                        except Exception as exc:notify_error(exc)
                    ui.button('Разрешить план (я рядом с роботом)',on_click=authorize_trial)
                with ui.row():
                    ui.button('Посмотреть запись',on_click=view_selected)

                ui.label('Повтор и политика требуют отдельного разрешения выбранного плана. Во время ожидания расчёта Tank стоит. Checkpoint можно проверить без движения во вкладке «Обучение».').classes('muted')
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
        options={k:v.get('label',k) for k,v in PROFILES.items() if v['robot']==robot}
        task.set_options(options,value=next(iter(options)))
    last_frame=[None]
    def refresh():
        if PROFILES[task.value]['robot']!=control.robot:
            options={k:v.get('label',k) for k,v in PROFILES.items() if v['robot']==control.robot}
            task.set_options(options,value=next(iter(options)))
        authorized=control.authorized()
        tank_bench=control.robot=='tank' and bench_authorized(DATA/'configs')
        trial_start_button.set_enabled(authorized and local_work.hardware_allowed(control.robot) and bool(prepared_trial[0]))
        supported=control.robot in {'so101','tank'}
        idle=not control.maintenance and not (session.thread and session.thread.is_alive())
        start_button.set_enabled(authorized and supported and idle and not tank_bench)
        take_button.set_enabled(supported and idle and (authorized or control.robot=='so101'))
        teleop_button.set_enabled(authorized and supported and idle)
        tank_start.set_enabled(authorized and idle)
        tank_start.set_text('Подключается…' if session.phase=='preparing' and control.robot=='tank' else 'Включить управление')
        old_button.set_enabled(supported and idle)
        tank_restore.set_enabled(idle)
        ownership_title.set_visibility(control.robot=='so101')
        ownership_controls.set_visibility(control.robot=='so101')
        ownership_help.set_visibility(control.robot=='so101')
        device_table.set_visibility(control.robot=='so101')
        tank_settings.set_visibility(control.robot=='tank')
        teleop_button.set_text('Включить проверку на подставке' if tank_bench else ('Включить управление' if control.robot=='tank' else 'Проверить телеперацию'))
        tank_help.set_text('Только на подставке. Включите управление и дождитесь готовности.' if tank_bench else 'Нажмите «Включить управление», дождитесь готовности и удерживайте стрелки. Отпускание останавливает движение. Для показа откройте «Запись».')
        tank_controls.set_visibility(control.robot=='tank')
        pilot_options.set_visibility(control.robot=='tank')
        pilot_active=bool(session.pilot and session.thread and session.thread.is_alive())
        tank_pilot_state.set_visibility(pilot_active)
        correction_controls.set_visibility(pilot_active)
        if pilot_active:
            ps=session.pilot.status()
            label={'policy':'Действует политика · решение о помощи ещё не принято',
                   'hold':'Tank стоит · ожидание коррекции',
                   'correction':'Ваша коррекция · управляйте стрелками',
                   'resumed':'Коррекция завершена · действует политика'}[ps['phase']]
            if ps['assigned']==0:label='Назначено без помощи · действует политика'
            if ps['phase'] in {'hold','correction'}:label+=f" · осталось {ps['remaining_s']:.1f} с"
            tank_pilot_state.set_text(label)
        rosmaster_panel.set_visibility(control.robot=='rosmaster')
        for button in (take_button,teleop_button,old_button):button.set_visibility(control.robot!='rosmaster')
        wanted='/rosmaster/control' if control.robot=='rosmaster' else 'about:blank'
        if rosmaster_frame.props.get('src')!=wanted:
            rosmaster_frame.props(f'src="{wanted}"')
        task_title.set_text('Подъехать к видимой цели → остановиться в зоне' if control.robot=='tank' else 'Кубик → коробка → отвести руку')
        task_description.set_text({'so101':'Wrist RGB и измеренные суставы; цель — отправленная команда Follower.',
            'tank':'Передняя RGB; состояние — предыдущая команда, измеренной скорости колёс нет.',
            'rosmaster':'ROSMASTER записывается в панели на вкладке «Управление» → карточка «Запись показа». Подъезд: камера + измеренная скорость колёс → команда базы. Кубик: камера + измеренные 6 суставов → цель руки. 20 Гц.'}[control.robot])
        selected.set_text('Выбран: '+ROBOT_NAMES[control.robot])
        smol_pass=(DATA/'validation/smolvla-pass.json').exists()
        smol_button.set_enabled(smol_pass and task.value=='so101_pick_place')
        smol_full_button.set_enabled(smol_pass and task.value=='so101_pick_place')
        collection_info.set_text(local_pages.collection_status(task.value))
        for ident,badge in fleet_labels.items():
            badge.set_text('Отдельная страница управления' if ident=='rosmaster' and rosmaster_control.enabled() else
                ('Не опрашивается · локальный режим' if not local_work.hardware_allowed(ident) else ('Выбран' if ident==control.robot else 'Не выбран')))
        act_pass=(DATA/'validation/act-pass.json').exists()
        ready.set_text('Программный smoke: ACT '+('PASS' if act_pass else 'не проверен')+' · SmolVLA '+('PASS' if smol_pass else 'не проверен')+' · физические навыки не проверены')
        if current_stage()=='B' and local_work.hardware_allowed('rosmaster'):
            stage_label.set_text('Этап B · ROSMASTER. Камера и измерения доступны в его панели; движение — только явной кнопкой рядом с роботом. Tank и SO-101 пока отключены от Explorer.')
        elif tank_bench:
            stage_label.set_text('Tank · проверка на подставке · штатная скорость · движение на полу и запись показов пока не приняты')
        elif control.robot=='tank' and authorized:
            stage_label.set_text('Tank · ручное управление и запись доступны · отпускание и STOP подтверждены · автономный навык ещё не проверен')
        elif not local_work.hardware_allowed(control.robot):
            stage_label.set_text('Сбор и испытания остаются в локальном режиме. Ручное управление ROSMASTER доступно по отдельной ссылке в его карточке.' if rosmaster_control.enabled() else 'Локальный режим · роботы не опрашиваются. Датасеты, обучение и анализ доступны без включения устройств.')
        elif control.robot=='tank' and (DATA/'validation/tank-motion-incident.json').exists():
            stage_label.set_text('Tank: движение заблокировано после отказа STOP. Диагностика; физическая приёмка не пройдена.')
        elif current_stage()=='C':
            stage_label.set_text('Этап C · подключение Tank. Приёмка SO-101 завершена; физический навык не проверен.')
        elif authorized:
            stage_label.set_text('Этап B · разрешённый физический блок. Движение начинается только явной кнопкой.')
        phases={'idle':'Остановлено','preparing':'Подготовка — пока не двигайте Leader','aligning':'Плавное совмещение — держите Leader неподвижно','following':'Следование — можно двигать Leader','error':'Ошибка','saving':'Сохранение эпизода — дождитесь завершения','trial':'Испытание — действует STOP, исход отмечается оператором'}
        if control.robot=='tank':
            phases.update(preparing='Подключение камеры и шасси — дождитесь готовности',aligning='Подготовка',following='Управление готово — можно удерживать стрелки',
                          paused='Видео задерживается — движение остановлено. После появления видео отпустите и снова нажмите стрелку.')
        if control.robot=='tank' and session.phase=='idle':
            phases['idle']='Управление выключено — нажмите «Включить управление»'
        phase_text='Передача управления — дождитесь завершения' if control.maintenance else phases.get(session.phase,session.phase)
        recording_status.set_text(session.error or (phase_text+' · кадров: '+str(session.frame_count)+((' · '+control.last_stop_reason) if control.last_stop_reason else '')))
        device_table.rows=[{'name':k,'state':('занято' if v['owners'] else 'USB обнаружено, не открыто') if v['present'] else 'отсутствует'} for k,v in inventory().items()]
        device_table.update()
        profile_table.rows=local_pages.profile_statuses();profile_table.update()
        episode_table.rows=catalog.rows('episodes');episode_table.update()
        job_table.rows=catalog.rows('jobs');job_table.update()
        event_table.rows=catalog.rows('events')[:20];event_table.update()
        mode_label.set_text((session.error or phase_text) if control.robot=='tank' else control.mode+' · '+(session.error or phase_text))
        pause_reason=(session.latest or {}).get('pause_reason')
        tank_motion_state.set_text((session.error or pause_reason or phase_text).removeprefix('RuntimeError: '))
        tank_motion_state.classes('text-red-4' if session.error else 'text-green-4',remove='text-red-4 text-green-4')
        if control.robot=='tank' and not tank_bench and (DATA/'validation/tank-motion-incident.json').exists():
            tank_motion_state.set_text('Tank заблокирован после отказа STOP. Соединение с моторной платой пропало; закрытие программы не подтверждает остановку гусениц. Повторный запуск запрещён до проверки причины.')
            tank_motion_state.classes('text-red-4',remove='text-green-4')
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
        if not local_work.hardware_allowed('tank') and not (control.robot=='tank' and bench_authorized(DATA/'configs')):
            tank_status.set_text('Сетевой опрос Tank отключён в локальном режиме')
        elif control.robot=='tank':
            def probe():
                with tank.client() as http:
                    response=http.get('/health');response.raise_for_status();return response.json()
            try:
                observed=await asyncio.to_thread(probe)
                link=observed.get('motor_link',{})
                tank_link_controls.set_visibility(bool(link.get('pending')))
                if link.get('pending'):
                    stamp=link.get('last_failure_at')
                    when=time.strftime('%H:%M:%S',time.localtime(stamp)) if stamp else 'неизвестное время'
                    warning=' Частые отказы USB: проверьте кабель и питание.' if link.get('frequent_failures') else ''
                    tank_link_notice.set_text('Связь с моторной платой прервалась в '+when+'. Убедитесь, что Tank стоит, затем подтвердите продолжение.'+warning)
                voltage=observed['battery_voltage']
                fresh=observed['battery_received_ns'] is not None and observed['monotonic_ns']-observed['battery_received_ns']<2_000_000_000
                battery=f' · батарея {voltage:.2f} В' if fresh else ''
                tank_status.set_text('Tank подключён по Wi-Fi · '+('управление включено' if observed['armed'] else 'управление выключено')+battery)
            except Exception as exc:tank_status.set_text('Tank недоступен: '+str(exc))
    refresh()
    ui.timer(2,refresh_tank)
    ui.timer(2,refresh)
    ui.timer(.25,refresh_preview)

async def prepare_recording_runtime():
    from .recording import prepare_runtime
    await asyncio.to_thread(prepare_runtime)
    catalog.event('info','Записывающий модуль готов; аппаратура не открывалась')

app.on_startup(prepare_recording_runtime)
app.on_startup(lambda:catalog.event('info','Explorer 3.0 запущен без открытия аппаратуры'))
app.on_shutdown(lambda:control.stop('Служба останавливается'))
app.on_shutdown(lambda:jobs.cancel())
from . import sim_service
app.on_shutdown(sim_service.shutdown)
from .fleet_service import service as fleet_service
app.on_shutdown(fleet_service.shutdown)
app.timer(.1,control.watchdog)
app.timer(2,jobs.tick)
ui.run(host=os.environ.get('EXPLORER3_HOST','127.0.0.1'),port=int(os.environ.get('EXPLORER3_PORT','8030')),title='Explorer 3.0',
       reload=False,show=False,tailwind=False,language='ru',favicon='🧭',timeout_graceful_shutdown=3)
