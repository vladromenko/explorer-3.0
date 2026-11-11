"""Explorer 3.0 — one NiceGUI/FastAPI server, local assets, inert startup."""
import json
import os
from pathlib import Path
from nicegui import app,ui
from . import catalog
from .control import Control
from .jobs import Jobs
from .session import Session
from . import handoff
from .network import LocalNetworkOnly
from . import viewer
from .paths import DATA, ROOT, prepare
from .profiles import PROFILES
from .resources import owners

prepare()
app.add_middleware(LocalNetworkOnly)
catalog.initialize()
control=Control()
jobs=Jobs(control)
session=Session(control)
ROBOT_NAMES={'so101':'SO-101','tank':'Tank','rosmaster':'ROSMASTER'}
DEVICE_PATHS={'leader':'/dev/so101-leader','follower':'/dev/so101-follower',
              'wrist':'/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0'}

def inventory():
    held=owners(list(DEVICE_PATHS.values()))
    return {k:{'present':Path(v).exists(),'owners':[o for o in held if str(Path(v).resolve()) in o['devices']]}
            for k,v in DEVICE_PATHS.items()}

@app.get('/api/health')
def health():
    return {'product':'Explorer 3.0','stage':'A' if not control.authorized() else 'B','armed':False if control.owner is None else True,
            'selected_robot':control.robot,'mode':control.mode,'devices':inventory(),'data_root':str(DATA),
            'session':{'phase':session.phase,'frames':session.frame_count,'error':session.error,
                       'recording':session.record_requested,'thread_active':bool(session.thread and session.thread.is_alive())}}

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
    def lost():
        if control.owner==owner:
            session.finish('aborted')
    ui.on('operator_lost',lost)
    client.on_disconnect(lost)
    ui.add_body_html("<script>setInterval(()=>{if(!document.hidden) emitEvent('operator_heartbeat')},250);document.addEventListener('visibilitychange',()=>{if(document.hidden) emitEvent('operator_lost')});window.addEventListener('blur',()=>emitEvent('operator_lost'));window.addEventListener('pagehide',()=>emitEvent('operator_lost'));</script>")
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
    with ui.element('div').classes('fleet'):
        for ident,name in ROBOT_NAMES.items():
            with ui.card().classes('robot'):
                ui.label(name).style('font-size:22px;font-weight:650')
                label='USB обнаружены · требуется приёмка' if ident=='so101' and all(v['present'] for v in inventory().values()) else 'Подключение ещё не проверено'
                ui.badge(label,color='grey-8')
                with ui.element('div').classes('preview'):
                    ui.label('Камера не открыта' if ident=='so101' else 'Подключение на этапе '+('C' if ident=='tank' else 'D'))
                ui.button('Выбрать '+name,on_click=lambda robot=ident:select(robot)).props('outline')
    with ui.element('div').classes('workspace'):
        selected=ui.label('Выбран: '+ROBOT_NAMES[control.robot]).classes('selected')
        mode_label=ui.label('Наблюдение · приводы не включены').classes('muted')
        task=ui.select({k:k for k,v in PROFILES.items() if v['robot']==control.robot},value='so101_pick_place',label='Задача').style('min-width:320px')
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
                    def take_control():
                        try:
                            handoff.enter(control,list(DEVICE_PATHS.values()))
                            ui.notify('Ресурсы освобождены; движение не началось')
                        except Exception as exc:
                            notify_error(exc)
                    def old_control():
                        try:
                            if session.thread and session.thread.is_alive():
                                raise RuntimeError('Дождитесь закрытия аппаратной сессии после STOP')
                            handoff.restore(control)
                            ui.notify('Прежние службы восстановлены без команды движения')
                        except Exception as exc:
                            notify_error(exc)
                    take_button=ui.button('Explorer 3.0',on_click=take_control)
                    def teleop_check():
                        try:
                            session.start(owner,'so101_pick_place','teleop-check',record=False)
                        except Exception as exc:
                            notify_error(exc)
                    teleop_button=ui.button('Проверить телеперацию',on_click=teleop_check)
                    old_button=ui.button('Прежнее управление',on_click=old_control)
                ui.label('Проверка телеперации не записывает показ. После проверки нажмите STOP.').classes('muted')
                device_table=ui.table(columns=[{'name':'name','label':'Устройство','field':'name'}, {'name':'state','label':'Фактическое состояние','field':'state'}],
                    rows=[],row_key='name')
                ui.label('SO-101 управляется Leader. Tank и ROSMASTER подключаются по очереди.').classes('muted')
            with ui.tab_panel(record):
                ui.label('Кубик → коробка → отвести руку').style('font-size:20px')
                ui.label('Только wrist RGB и измеренные суставы; цель — отправленная команда Follower после ограничений.').classes('muted')
                scene=ui.input('Расстановка',value='scene-01')
                def begin():
                    try:
                        session.start(owner,task.value,scene.value)
                        ui.notify('Запуск разрешённой сессии')
                    except Exception as exc:
                        notify_error(exc)
                with ui.row():
                    start_button=ui.button('Начать показ',on_click=begin)
                    ui.button('Закончить · успех',on_click=lambda:session.finish('success'))
                    ui.button('Закончить · неуспех',on_click=lambda:session.finish('failure'))
                    ui.button('Прервать',on_click=lambda:session.finish('aborted'))
                recording_status=ui.label('Нет активной записи')
                preview=ui.image().style('max-width:640px')
                ui.label('Для активации необходимы разрешение на физический блок B и проверка телеперации.').classes('notice')
                chart=ui.echart({'backgroundColor':'transparent','textStyle':{'color':'#9aacbf'},'tooltip':{'trigger':'axis'},
                    'legend':{'data':['Команда','Измерение'],'textStyle':{'color':'#9aacbf'}},
                    'xAxis':{'type':'value','name':'секунды'},'yAxis':{'type':'value'},
                    'series':[{'name':'Команда','type':'line','data':[]},{'name':'Измерение','type':'line','data':[]}]}).style('height:240px;width:100%')
            with ui.tab_panel(datasets):
                ui.label('Исходные эпизоды сохраняются. Исключение влияет только на подбор обучения.').classes('muted')
                episode_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('id','Эпизод'),('profile','Задача'),('outcome','Исход'),('frames','Кадры'),('excluded','Исключён')]],rows=[],row_key='id',selection='single')
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
            with ui.tab_panel(training):
                ui.label('Один trainer · отдельная политика для каждой задачи').style('font-size:19px')
                ui.label('ACT: upstream LeRobot 0.6.1. Smoke: 50 шагов; это проверка тракта, не навык.').classes('muted')
                def enqueue():
                    try:
                        ident=jobs.enqueue(task.value,'act',smoke=True)
                        ui.notify('Задание '+ident+' поставлено в очередь')
                    except Exception as exc:
                        notify_error(exc)
                with ui.row():
                    ui.button('ACT · проверить на 3 показах',on_click=enqueue)
                    def smol():
                        try:
                            jobs.enqueue(task.value,'smolvla',smoke=True)
                            ui.notify('SmolVLA поставлена в очередь')
                        except Exception as exc:
                            notify_error(exc)
                    smol_button=ui.button('SmolVLA · smoke',on_click=smol)
                    ui.button('Остановить trainer',on_click=jobs.cancel).props('outline')
                job_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('id','Задание'),('profile','Задача'),('policy','Модель'),('status','Состояние'),('error','Ошибка')]],rows=[],row_key='id')
            with ui.tab_panel(trials):
                ui.label('Программная готовность ≠ checkpoint ≠ физический навык').style('font-size:19px')
                with ui.row():
                    ui.button('Посмотреть запись').disable()
                    ui.button('Повторить команды на роботе').disable()
                    ui.button('Испытать политику').disable()
                ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('profile','Задача'),('record','Запись'),('train','Обучение'),('skill','Физический навык')]],
                    rows=[{'profile':k,'record':'не проверено','train':'не проверено','skill':'не проверено'} for k in PROFILES])
        with ui.expansion('Журнал событий и ошибок',icon='receipt_long').style('width:100%'):
            event_table=ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('level','Уровень'),('message','Сообщение')]],rows=[],row_key='id')
    with ui.footer():
        ui.label('Homebrain · автономное локальное приложение · питание роботов: состояние неизвестно')
    def select(robot):
        control.select(robot)
        selected.set_text('Выбран: '+ROBOT_NAMES[robot])
        options={k:k for k,v in PROFILES.items() if v['robot']==robot}
        task.set_options(options,value=next(iter(options)))
    last_frame=[None]
    def refresh():
        authorized=control.authorized()
        start_button.set_enabled(authorized and control.robot=='so101' and not (session.thread and session.thread.is_alive()))
        take_button.set_enabled(authorized and control.robot=='so101')
        teleop_button.set_enabled(authorized and control.robot=='so101' and not (session.thread and session.thread.is_alive()))
        old_button.set_enabled(authorized)
        smol_pass=(DATA/'validation/smolvla-pass.json').exists()
        smol_button.set_enabled(smol_pass and task.value=='so101_pick_place')
        act_pass=(DATA/'validation/act-pass.json').exists()
        ready.set_text('Программный smoke: ACT '+('PASS' if act_pass else 'не проверен')+' · SmolVLA '+('PASS' if smol_pass else 'не проверен')+' · физические навыки не проверены')
        if authorized:
            stage_label.set_text('Этап B · разрешённый физический блок. Движение начинается только кнопкой записи.')
        phases={'idle':'Остановлено','preparing':'Подготовка — пока не двигайте Leader','aligning':'Плавное совмещение — держите Leader неподвижно','following':'Следование — можно двигать Leader','error':'Ошибка','saving':'Сохранение эпизода — дождитесь завершения'}
        phase_text=phases.get(session.phase,session.phase)
        recording_status.set_text(session.error or (phase_text+' · кадров: '+str(session.frame_count)))
        device_table.rows=[{'name':k,'state':('занято' if v['owners'] else 'USB обнаружено, не открыто') if v['present'] else 'отсутствует'} for k,v in inventory().items()]
        device_table.update()
        episode_table.rows=catalog.rows('episodes');episode_table.update()
        job_table.rows=catalog.rows('jobs');job_table.update()
        event_table.rows=catalog.rows('events')[:20];event_table.update()
        mode_label.set_text(control.mode+' · '+phase_text)
    def refresh_preview():
        with session.lock:
            sample=session.latest
        if sample and sample['frame_id']!=last_frame[0]:
            import cv2,base64
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(sample['images']['wrist']['image'],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
            if ok:
                preview.set_source('data:image/jpeg;base64,'+base64.b64encode(jpeg).decode())
            for series,key in zip(chart.options['series'],('action','observation.state')):
                series['data'].append([sample['frame_id']/30,sample[key][0]])
                series['data']=series['data'][-300:]
            chart.update()
            last_frame[0]=sample['frame_id']
    refresh()
    ui.timer(2,refresh)
    ui.timer(1/15,refresh_preview)

app.on_startup(lambda:catalog.event('info','Explorer 3.0 запущен без открытия аппаратуры'))
app.on_shutdown(lambda:control.stop('Служба останавливается'))
app.on_shutdown(jobs.cancel)
app.timer(.1,control.watchdog)
app.timer(2,jobs.tick)
ui.run(host=os.environ.get('EXPLORER3_HOST','127.0.0.1'),port=int(os.environ.get('EXPLORER3_PORT','8030')),title='Explorer 3.0',
       reload=False,show=False,tailwind=False,language='ru',favicon='🧭')
