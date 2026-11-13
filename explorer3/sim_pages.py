"""MuJoCo workspace in the same Explorer visual system and compute queue."""
import asyncio
import uuid
from fastapi import Request
from nicegui import ui
from . import sim_service,catalog
from .sim.drive import command as drive_command

STYLE='''
/* Explicit utilities: this offline UI runs with Tailwind disabled. */
.w-full {width:100%}
.sim-panel .q-field {width:100%;min-width:0}
.sim-panel {box-sizing:border-box}
.sim-layout { display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:18px;width:100%;align-items:start }
.sim-panel {background:#192331;border:1px solid #344253;border-radius:14px;padding:20px;width:100%;gap:14px;box-shadow:none}
.sim-panel .q-img {width:100%;border-radius:10px}
.sim-controls {display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;width:100%}
.sim-kicker {color:#7dd3c7;font-size:12px;letter-spacing:.1em;text-transform:uppercase}
.sim-title {font-size:21px;font-weight:650}
.sim-badge {border:1px solid #41796f;border-radius:20px;padding:6px 12px;background:#163b35;color:#9de7dd;font-weight:600}
.sim-viewer {width:100%;height:560px;border:0;border-radius:10px;background:#111923}
.sim-status {color:#c5d8e2;min-height:24px}
@media(max-width:950px){.sim-layout{grid-template-columns:1fr}.sim-viewer{height:420px}}
'''


def install(stop_real,css,jobs):
    @ui.page('/sim')
    async def page(request:Request):
        await stop_real()
        ui.add_css(css+STYLE)
        ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        ui.dark_mode().enable()
        active=False;busy=False;started=False;displayed_profile=None;last_episode=None;local_error=None
        client=ui.context.client
        operator=uuid.uuid4().hex
        controls=[];drive_inputs={}
        try:initial_state=await asyncio.to_thread(sim_service.request)
        except Exception:initial_state=None
        initial_profile=(initial_state or {}).get('profile','so101_pick_place_sim')
        control_schema={'limits':[[-110,-100,-96.8,-95,-157,0],[110,100,96.8,95,162,100]],'target':[0]*6}
        if initial_state:control_schema.update(limits=initial_state['limits'],target=initial_state['target'])
        async def call(command=None,**kwargs):
            nonlocal local_error
            try:
                result=await asyncio.to_thread(sim_service.request,command,operator=operator,**kwargs)
                if command not in (None,'heartbeat'):local_error=None
                return result
            except Exception as exc:local_error=str(exc);status.set_text(local_error);return None
        async def pause():
            nonlocal active
            active=False;drive_inputs.clear()
            if started:await call('pause')
        async def release():
            nonlocal active
            active=False;drive_inputs.clear()
            if started:await call('release')
        async def claim():
            if not started:await launch()
            return started and bool(await call('claim'))
        async def launch():
            nonlocal started
            try:
                result=await asyncio.to_thread(sim_service.start)
                started=True
                control_schema.update(limits=result['limits'],target=result['target'])
                if result['profile']==profile_select.value:controls_panel.refresh()
                studio.props(f'src=http://{request.url.hostname}:{sim_service.VIEWER_PORT}/')
                status.set_text('Сцена готова · пауза')
            except Exception as exc:status.set_text(str(exc))
        async def play():
            nonlocal active
            if not started:await launch()
            active=await claim()
        async def demonstrate():
            nonlocal active
            active=False;drive_inputs.clear()
            if await claim():await call('demonstrate',seed=int(seed.value),variation=variation_select.value)
        async def reset():
            await release()
            if await claim() and await call('reset',seed=int(seed.value),variation=variation_select.value):
                state=await call()
                control_schema.update(limits=state['limits'],target=state['target'])
                controls_panel.refresh()
        async def select_profile():
            await release()
            if not started:await launch()
            if await claim():await call('profile',profile=profile_select.value)
            history.refresh()
        async def record():
            if not started:await launch()
            if await claim():await call('record')
        async def finish():
            nonlocal active
            active=False;drive_inputs.clear()
            if await claim():await call('finish')
        async def enqueue(kind):
            try:
                state=await call()
                if not state or state['profile']!=profile_select.value:
                    raise ValueError('Сначала откройте выбранную сцену')
                if kind=='ppo':ident=await asyncio.to_thread(jobs.sim_ppo,profile_select.value,
                    sim_model_hash=state['task']['model']['xml_sha256'],variation=variation_select.value)
                else:ident=await asyncio.to_thread(jobs.enqueue,profile_select.value,policy_select.value,smoke=kind!='act_full',
                    sim_model_hash=state['task']['model']['xml_sha256'])
                job_status.set_text('В очереди: '+ident)
            except Exception as exc:job_status.set_text(str(exc))

        with ui.element('div').classes('hero'):
            with ui.column().style('gap:3px'):
                ui.label('HOMEBRAIN / ЛОКАЛЬНАЯ ЛАБОРАТОРИЯ').classes('kicker')
                ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Модели, показы и обучение без подключения роботов').classes('muted')
            with ui.row().classes('items-center'):
                ui.label('СИМУЛЯЦИЯ · MuJoCo').classes('sim-badge')
                ui.button('STOP',icon='stop_circle',on_click=pause,color='negative').props('size=lg')
        with ui.row().classes('w-full items-center justify-between'):
            ui.link('← Реальные роботы','/').classes('muted')
            ui.link('Общий стенд · три робота →','/sim/fleet')
            ui.link('Зрение · YOLOE / SAM 3','/sim/vision')
            ui.link('Сравнить помощь','/sim/branches')
            profile_select=ui.select({'so101_pick_place_sim':'SO-101 · Menagerie','rosmaster_pick_place_sim':'ROSMASTER · рука','rosmaster_dock_sim':'ROSMASTER · шасси','tank_dock_sim':'Tank · шасси'},
                value=initial_profile,label='Робот в симуляции',on_change=select_profile).style('min-width:300px')
            variation_select=ui.select({'nominal':'Базовая сцена · сетка 10 см','varied':'Вариации для обучения'},
                value=(initial_state or {}).get('task',{}).get('variation',{}).get('mode','nominal'),
                label='Условия сцены',on_change=reset).style('min-width:280px')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                with ui.row().classes('w-full items-center justify-between'):
                    task_title=ui.label('Кубик в коробку').classes('sim-title')
                    ui.label('Физическая модель · калибровка впереди').classes('muted')
                studio=ui.element('iframe').props('title="MuJoCo Studio"').classes('sim-viewer')
                with ui.row().classes('items-center w-full'):
                    ui.button('Открыть сцену',icon='view_in_ar',on_click=launch)
                    ui.button('Двигать модель',icon='play_arrow',on_click=play)
                    ui.button('Показать задачу · без обучения',icon='science',on_click=demonstrate).props('outline')
                    ui.button('Пауза / STOP',icon='pause',on_click=pause,color='secondary')
                status=ui.label('Откройте сцену, чтобы начать').classes('sim-status')
                ui.label('Клетка пола — 10 см. Расчёт на Homebrain; мышью можно вращать и приближать сцену.').classes('muted text-xs')
                ui.label('Вариации меняют фон, свет, небольшое смещение камеры и параметры кубика. Диапазоны пока проверочные, без калибровки по вашему роботу.').classes('muted text-xs')
                model_note=ui.label('').classes('muted text-xs')
            with ui.column().classes('sim-panel'):
                camera_title=ui.label('Камера руки').classes('sim-title')
                image=ui.image().props('no-transition no-spinner').classes('w-full rounded-lg')
                ui.label('Это изображение получает визуальная политика').classes('muted text-xs')
                ui.separator()
                ui.label('Показ задачи').classes('sim-title')
                task_help=ui.label('Захватить кубик → положить в коробку → отпустить → отвести руку.').classes('muted')
                with ui.row():
                    ui.button('Начать запись',icon='fiber_manual_record',on_click=record)
                    ui.button('Закончить запись',icon='done',on_click=finish).props('outline')
                success_help=ui.label('Успех определяется физикой: кубик отпущен и неподвижен в коробке 0,5 секунды.').classes('muted text-xs')
                with ui.row().classes('items-end'):
                    seed=ui.number('Расстановка',value=0,min=0,precision=0).style('width:130px')
                    ui.button('Сбросить сцену',icon='restart_alt',on_click=reset).props('flat')
        async def drive(key,pressed,source=None):
            nonlocal active
            if not profile_select.value.endswith('_dock_sim'):return
            if pressed:
                if not started:await launch()
                if not active and not await claim():return
                drive_inputs[source or key]=key;active=started
            else:drive_inputs.pop(source or key,None)
            if started:await call('target',target=drive_command(profile_select.value,drive_inputs.values()))
        @ui.refreshable
        def controls_panel():
            controls.clear()
            with ui.column().classes('sim-panel'):
                if profile_select.value.endswith('_dock_sim'):
                    ui.label('Движение шасси').classes('sim-title')
                    ui.label('Удерживайте кнопку или W/A/S/D. Отпускание — нулевая команда; пробел — пауза.').classes('muted')
                    with ui.element('div').style('display:grid;grid-template-columns:repeat(3,72px);gap:8px'):
                        for label,key in [('↖','up_left'),('↑','up'),('↗','up_right'),('←','left'),('STOP','stop'),('→','right'),('↙','down_left'),('↓','down'),('↘','down_right')]:
                            button=ui.button(label,color='negative' if key=='stop' else 'primary').style('height:48px')
                            if key=='stop':button.on_click(pause)
                            else:
                                button.on('pointerdown',lambda key=key:drive(key,True))
                                button.on('pointerup',lambda key=key:drive(key,False))
                                button.on('pointerleave',lambda key=key:drive(key,False))
                                button.on('pointercancel',lambda key=key:drive(key,False))
                    if profile_select.value=='rosmaster_dock_sim':
                        ui.label('←/→ — движение боком; Q/E — поворот на месте.').classes('muted')
                        with ui.row():
                            for text,key in [('↶ Поворот влево','turnleft'),('↷ Поворот вправо','turnright')]:
                                button=ui.button(text)
                                button.on('pointerdown',lambda key=key:drive(key,True))
                                button.on('pointerup',lambda key=key:drive(key,False))
                                button.on('pointerleave',lambda key=key:drive(key,False))
                else:
                    ui.label('Управление суставами').classes('sim-title')
                    ui.label('Задайте положение и нажмите «Двигать модель». Скрытие страницы ставит сцену на паузу.').classes('muted')
                    names=['Основание · °','Плечо · °','Локоть · °','Наклон кисти · °','Вращение кисти · °','Захват · %' if profile_select.value.startswith('so101') else 'Захват · °']
                    with ui.element('div').classes('sim-controls'):
                        for i,name in enumerate(names):
                            with ui.column().classes('w-full').style('gap:3px'):
                                ui.label(name).classes('muted')
                                controls.append(ui.slider(min=float(control_schema['limits'][0][i]),max=float(control_schema['limits'][1][i]),
                                    value=round(control_schema['target'][i],1),step=.5).props('label').classes('w-full'))
        controls_panel()
        async def keyboard(event):
            args=event.args
            await drive(args['key'],args['pressed'],args['source'])
        ui.on('sim_drive_key',keyboard)
        ui.on('sim_stop',pause)
        ui.add_body_html("""<script>
        const simKeys={KeyW:'up',KeyS:'down',KeyA:'left',KeyD:'right',ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',KeyQ:'turnleft',KeyE:'turnright'};
        function simKey(e,pressed){if(['INPUT','TEXTAREA'].includes(e.target.tagName))return;
          if(e.code==='Space'){e.preventDefault();emitEvent('sim_stop');return;}
          const key=simKeys[e.code];if(key){e.preventDefault();if(!e.repeat)emitEvent('sim_drive_key',{key,pressed,source:e.code});}}
        window.addEventListener('keydown',e=>simKey(e,true));window.addEventListener('keyup',e=>simKey(e,false));
        </script>""")
        with ui.column().classes('sim-panel'):
            ui.label('Обучение и результаты').classes('sim-title')
            ui.link('Подготовить свой бюджет или дообучение →','/sim/training')
            ui.label('ACT и Diffusion используют камеру и состояние; SmolVLA также получает текст задачи. PPO использует полное состояние симуляции. Обучение запускается только вашей кнопкой.').classes('muted')
            from .policy_catalog import POLICIES
            policy_select=ui.select(POLICIES,value='act',label='Модель управления').classes('w-full')
            with ui.row():
                ui.button('Проверить тракт · 50 шагов',icon='model_training',on_click=lambda:enqueue('act')).props('outline')
                ui.button('Обучить выбранную модель · 2 000 шагов',icon='model_training',on_click=lambda:enqueue('act_full')).props('outline')
                ui.button('Обучить PPO · 16 384 шага',icon='model_training',on_click=lambda:enqueue('ppo')).props('outline')
                ui.button('Обновить список',icon='refresh',on_click=lambda:history.refresh()).props('flat')
            job_status=ui.label('Один trainer за раз. 50 шагов проверяют тракт, а не готовность навыка.').classes('muted')
        @ui.refreshable
        def history():
            with ui.column().classes('sim-panel'):
                ui.label('Последние записи и задания').classes('sim-title')
                episodes=[e for e in catalog.rows('episodes') if e['profile']==profile_select.value][:5]
                tasks=[j for j in catalog.rows('jobs') if j['profile']==profile_select.value][:5]
                for episode in episodes:
                    ui.link(f"Показ {episode['id'][:8]} · {episode['outcome']} · {episode['frames']} кадров",f"/episodes/{episode['id']}")
                for task in tasks:
                    if task['policy'] in {'act','smolvla','diffusion'} and task['status']=='completed' and task['checkpoint']:
                        async def evaluate(ident=task['id']):
                            try:
                                created=await asyncio.to_thread(jobs.sim_evaluate,ident)
                                job_status.set_text('Испытание модели: '+created)
                            except Exception as exc:job_status.set_text(str(exc))
                        ui.button('Испытать '+task['policy'].upper()+' · '+task['id'][:8],on_click=evaluate).props('outline')
                        async def prepare_support(ident=task['id']):
                            try:
                                created=await asyncio.to_thread(jobs.prepare_support,ident)
                                job_status.set_text('Подготовка оценки помощи: '+created)
                            except Exception as exc:job_status.set_text(str(exc))
                        if task['policy']=='act':ui.button('Подготовить оценку помощи',on_click=prepare_support).props('outline')
                    ui.link(f"{task['policy'].upper()} · {task['status']} · {task['id'][:8]}",f"/jobs/{task['id']}")
                if not episodes and not tasks:ui.label('Здесь появятся сохранённые записи и запуски обучения').classes('muted')
        history()
        ui.on('sim_hidden',release)
        client.on_disconnect(release)
        ui.add_body_html("<script>document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('sim_hidden')});window.addEventListener('pagehide',()=>emitEvent('sim_hidden'));</script>")
        async def tick():
            nonlocal busy,active,displayed_profile,last_episode
            if not started or busy:return
            busy=True
            try:
                if active:
                    target=drive_command(profile_select.value,drive_inputs.values()) if profile_select.value.endswith('_dock_sim') else [float(c.value) for c in controls]
                    if not await call('target',target=target):active=False
                state=await call()
                if state:
                    if state.get('owner')==operator:await call('heartbeat')
                    if displayed_profile!=state['profile'] and state['profile']==profile_select.value:
                        displayed_profile=state['profile']
                        control_schema.update(limits=state['limits'],target=state['target']);controls_panel.refresh()
                        mobile=state['profile'].endswith('_dock_sim')
                        task_title.set_text('Подъехать к цели' if mobile else 'Кубик в коробку')
                        camera_title.set_text('Камера спереди' if mobile else 'Камера руки')
                        task_help.set_text('Подъехать к видимой цели → остановиться в зелёной зоне.' if mobile else 'Захватить кубик → положить в коробку → отпустить → отвести руку.')
                        success_help.set_text('Успех: положение, направление и малая скорость сохраняются в зоне 0,5 секунды.' if mobile else 'Кубик отпущен и неподвижен в коробке 0,5 секунды.')
                        model_note.set_text('MentorPi T1: геометрия совпадает с установленной моделью; параметры камеры измерены. Калибровка движения ещё впереди.' if state['profile']=='tank_dock_sim' else
                            'ROSMASTER M3Pro: исходная геометрия Yahboom, перенесённая из заводской модели.' if state['profile'].startswith('rosmaster') else
                            'SO-101: исходная модель MuJoCo Menagerie с контактной геометрией захвата.')
                    if state['episode']!=last_episode:
                        last_episode=state['episode'];history.refresh()
                    image.set_source('data:image/jpeg;base64,'+state['rgb'])
                    text='Движение модели' if not state['paused'] else 'Пауза'
                    if state.get('controller')=='scripted_expert':text='Проверка заданной траекторией · без обучения'
                    if state['recording']:text+=' · идёт запись'
                    if state['finalizing']:text+=' · сохранение'
                    if state['episode']:text+=' · запись сохранена: '+state['episode'][:8]
                    if state['outcome']:text+=' · результат: '+state['outcome'];active=False
                    status.set_text(local_error or state['error'] or text)
            finally:busy=False
        ui.timer(.2,tick)
