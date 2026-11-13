"""MuJoCo workspace in the same Explorer visual system and compute queue."""
import asyncio
from fastapi import Request
from nicegui import ui
from . import sim_service,catalog

STYLE='''
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
        active=False;busy=False;started=False
        client=ui.context.client
        controls=[]
        async def call(command=None,**kwargs):
            try:return await asyncio.to_thread(sim_service.request,command,**kwargs)
            except Exception as exc:status.set_text(str(exc));return None
        async def pause():
            nonlocal active
            active=False
            if started:await call('pause')
        async def launch():
            nonlocal started
            try:
                result=await asyncio.to_thread(sim_service.start)
                started=True
                for slider,value in zip(controls,result['target']):slider.set_value(round(value,1))
                studio.props(f'src=http://{request.url.hostname}:{sim_service.VIEWER_PORT}/')
                status.set_text('Сцена готова · пауза')
            except Exception as exc:status.set_text(str(exc))
        async def play():
            nonlocal active
            if not started:await launch()
            active=started
        async def reset():
            await pause()
            if await call('reset',seed=int(seed.value)):
                state=await call()
                for slider,value in zip(controls,state['target']):slider.set_value(round(value,1))
        async def record():
            if not started:await launch()
            if started:await call('record')
        async def finish():
            await pause();await call('finish')
        def enqueue(kind):
            try:
                ident=jobs.sim_ppo('so101_pick_place_sim') if kind=='ppo' else jobs.enqueue('so101_pick_place_sim','act',smoke=True)
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
            ui.label('SO-101 · Menagerie / Playground').classes('muted')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Кубик в коробку').classes('sim-title')
                    ui.label('Модель ещё не откалибрована').classes('muted')
                studio=ui.element('iframe').props('title="MuJoCo Studio"').classes('sim-viewer')
                with ui.row().classes('items-center w-full'):
                    ui.button('Открыть сцену',icon='view_in_ar',on_click=launch)
                    ui.button('Двигать модель',icon='play_arrow',on_click=play)
                    ui.button('Пауза / STOP',icon='pause',on_click=pause,color='secondary')
                status=ui.label('Откройте сцену, чтобы начать').classes('sim-status')
                ui.label('Физика работает на Homebrain. 3D-сцену можно вращать и приближать мышью.').classes('muted text-xs')
            with ui.column().classes('sim-panel'):
                ui.label('Камера на кисти').classes('sim-title')
                image=ui.image().props('no-transition no-spinner').classes('w-full rounded-lg')
                ui.label('Это изображение получает визуальная политика').classes('muted text-xs')
                ui.separator()
                ui.label('Показ задачи').classes('sim-title')
                ui.label('Захватить кубик → положить в коробку → отпустить → отвести руку.').classes('muted')
                with ui.row():
                    ui.button('Начать запись',icon='fiber_manual_record',on_click=record)
                    ui.button('Закончить запись',icon='done',on_click=finish).props('outline')
                ui.label('Успех определяется физикой: кубик отпущен и неподвижен в коробке 0,5 секунды.').classes('muted text-xs')
                with ui.row().classes('items-end'):
                    seed=ui.number('Расстановка',value=0,min=0,precision=0).style('width:130px')
                    ui.button('Сбросить сцену',icon='restart_alt',on_click=reset).props('flat')
        with ui.column().classes('sim-panel'):
            ui.label('Управление суставами').classes('sim-title')
            ui.label('Задайте положение ползунками и нажмите «Двигать модель». Скрытие страницы ставит сцену на паузу.').classes('muted')
            ranges=[(-110,110),(-100,100),(-96.8,96.8),(-95,95),(-157,162),(0,100)]
            names=['Основание · °','Плечо · °','Локоть · °','Наклон кисти · °','Вращение кисти · °','Захват · %']
            with ui.element('div').classes('sim-controls'):
                for name,(low,high) in zip(names,ranges):
                    with ui.column().classes('w-full').style('gap:3px'):
                        ui.label(name).classes('muted')
                        controls.append(ui.slider(min=low,max=high,value=(low+high)/2,step=.5).props('label').classes('w-full'))
        with ui.column().classes('sim-panel'):
            ui.label('Обучение и результаты').classes('sim-title')
            ui.label('ACT учится по камере и суставам. PPO использует полное состояние симуляции и остаётся симуляционным учителем.').classes('muted')
            with ui.row():
                ui.button('Проверить обучение ACT · 50 шагов',icon='model_training',on_click=lambda:enqueue('act')).props('outline')
                ui.button('Обучить PPO · 16 384 шага',icon='model_training',on_click=lambda:enqueue('ppo')).props('outline')
                ui.button('Обновить список',icon='refresh',on_click=lambda:history.refresh()).props('flat')
            job_status=ui.label('Один trainer за раз. 50 шагов проверяют тракт, а не готовность навыка.').classes('muted')
        @ui.refreshable
        def history():
            with ui.column().classes('sim-panel'):
                ui.label('Последние записи и задания').classes('sim-title')
                episodes=[e for e in catalog.rows('episodes') if e['profile']=='so101_pick_place_sim'][:5]
                tasks=[j for j in catalog.rows('jobs') if j['profile']=='so101_pick_place_sim'][:5]
                for episode in episodes:
                    ui.link(f"Показ {episode['id'][:8]} · {episode['outcome']} · {episode['frames']} кадров",f"/episodes/{episode['id']}")
                for task in tasks:
                    ui.link(f"{task['policy'].upper()} · {task['status']} · {task['id'][:8]}",f"/jobs/{task['id']}")
                if not episodes and not tasks:ui.label('Здесь появятся сохранённые записи и запуски обучения').classes('muted')
        history()
        ui.on('sim_hidden',pause)
        client.on_disconnect(pause)
        ui.add_body_html("<script>document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('sim_hidden')});window.addEventListener('pagehide',()=>emitEvent('sim_hidden'));</script>")
        async def tick():
            nonlocal busy,active
            if not started or busy:return
            busy=True
            try:
                if active:
                    if not await call('target',target=[float(c.value) for c in controls]):active=False
                state=await call()
                if state:
                    image.set_source('data:image/jpeg;base64,'+state['rgb'])
                    text='Движение модели' if not state['paused'] else 'Пауза'
                    if state['recording']:text+=' · идёт запись'
                    if state['finalizing']:text+=' · сохранение'
                    if state['episode']:text+=' · запись сохранена: '+state['episode'][:8]
                    if state['outcome']:text+=' · результат: '+state['outcome'];active=False
                    status.set_text(state['error'] or text)
            finally:busy=False
        ui.timer(.2,tick)
