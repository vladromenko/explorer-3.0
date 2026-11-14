"""Shared-world user workflow inside Explorer, using the native Studio viewer."""
import asyncio
import uuid
from fastapi import Request
from nicegui import ui
from .fleet_service import service
from .sim_pages import STYLE

NAMES={'so101':'SO-101','rosmaster':'ROSMASTER','tank':'Tank'}


def install(stop_real,css):
    @ui.page('/sim/fleet')
    async def page(request:Request):
        await stop_real();ui.add_css(css+STYLE);ui.dark_mode().enable()
        ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        operator=uuid.uuid4().hex;client=ui.context.client
        started=False;busy=False;active=False;latest=None;sliders=[];schema=None
        async def call(command=None,**kw):
            try:return await asyncio.to_thread(service.request,command,operator=operator,**kw)
            except Exception as exc:status.set_text(str(exc));return None
        async def launch():
            nonlocal started
            try:
                await asyncio.to_thread(service.start);started=True
                viewer.props(f'src=http://{request.url.hostname}:{service.viewer_port}/')
                status.set_text('Общий стенд готов · пауза')
            except Exception as exc:status.set_text(str(exc))
        async def claim():
            if not started:await launch()
            return started and bool(await call('claim'))
        async def pause():
            nonlocal active
            active=False
            if started:await call('pause')
        async def release():
            nonlocal active
            active=False
            if started:await call('release')
        async def select():
            await pause()
            if await claim():await call('select',agent=robot.value)
        async def demonstrate():
            nonlocal active
            active=False
            if await claim():await call('demonstrate',seed=int(seed.value))
        async def reset(single=False):
            nonlocal active
            active=False
            if await claim():await call('reset',seed=int(seed.value),agent=robot.value if single else None)
        async def change_scene():
            await pause()
            if await claim():await call('scene',scene=scene.value,seed=int(seed.value))
        async def simple(command):
            nonlocal active
            if command=='restore':active=False
            if await claim():await call(command)
        async def play():
            nonlocal active
            active=await claim()
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / ОБЩИЙ СТЕНД').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('SO-101 на столе · ROSMASTER и Tank на полу · один оператор').classes('muted')
                ui.label('Высота стола пока задана только для симуляции и не считается измеренной.').classes('muted')
            ui.button('STOP',icon='stop_circle',color='negative',on_click=pause).props('size=lg')
        with ui.row().classes('w-full justify-between items-center'):
            ui.link('← Отдельные задачи','/sim')
            ui.link('Зрение · YOLOE / SAM 3','/sim/vision')
            ui.link('Совместное исполнение политик →','/sim/fleet/policies')
            ui.label('СИМУЛЯЦИЯ · без подключения роботов').classes('sim-badge')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                ui.label('Общая лаборатория').classes('sim-title')
                viewer=ui.element('iframe').props('title="MuJoCo shared world"').classes('sim-viewer')
                with ui.row():
                    ui.button('Открыть общий стенд',icon='view_in_ar',on_click=launch)
                    ui.button('Показать три задачи · без обучения',icon='science',on_click=demonstrate).props('outline')
                    ui.button('Пауза / STOP',icon='pause',color='secondary',on_click=pause)
                status=ui.label('Откройте стенд').classes('sim-status')
                ui.label('Показ выполняет заданный геометрический алгоритм. Это проверка физики и программного тракта, не результат обучения.').classes('muted')
                ui.label('ROSMASTER выполняет задачу рукой с неподвижным шасси. Tank использует исходную геометрию T1; динамика ещё не откалибрована.').classes('muted text-xs')
            with ui.column().classes('sim-panel'):
                robot=ui.select(NAMES,value='so101',label='Робот оператора',on_change=select).classes('w-full')
                camera=ui.image().props('no-transition no-spinner')
                ui.label('Камера выбранного робота; общий вид не поступает политике').classes('muted text-xs')
                result=ui.label('').classes('sim-status')
                seed=ui.number('Расстановка',value=1,min=0,precision=0)
                from .sim.room import PRESETS
                scene=ui.select({k:v['label'] for k,v in PRESETS.items()},value='workshop',label='Окружение').classes('w-full')
                ui.button('Применить окружение · сброс в паузу',on_click=change_scene).props('outline')
                with ui.row():
                    ui.button('Сбросить выбранного',icon='restart_alt',on_click=lambda:reset(True)).props('outline')
                    ui.button('Сбросить всех',on_click=lambda:reset(False)).props('flat')
        @ui.refreshable
        def controls():
            sliders.clear()
            with ui.column().classes('sim-panel'):
                ui.label('Управление выбранным роботом').classes('sim-title')
                ui.label('Команды получает только выбранная машина. Переключение и скрытие страницы останавливают управление.').classes('muted')
                if latest:
                    entry=latest['world']['robots'][robot.value]
                    names=['Скорость · м/с','Поворот · рад/с'] if robot.value=='tank' else ['Основание · °','Плечо · °','Локоть · °','Наклон · °','Вращение · °','Захват · %' if robot.value=='so101' else 'Захват · °']
                    with ui.element('div').classes('sim-controls'):
                        for i,title in enumerate(names):
                            with ui.column().classes('w-full'):
                                ui.label(title)
                                sliders.append(ui.slider(min=entry['limits'][0][i],max=entry['limits'][1][i],value=float(entry['observation'][i]),step=.01 if robot.value=='tank' else .5).props('label').classes('w-full'))
                    ui.button('Двигать выбранного',icon='play_arrow',on_click=play)
                else:ui.label('Появится после открытия стенда').classes('muted')
        controls()
        with ui.column().classes('sim-panel'):
            ui.label('Состояние и очередь помощи').classes('sim-title')
            ui.label('Сохранение включает физику всех машин и очереди команд. Возврат в интерфейсе всегда ставит сцену на паузу.').classes('muted')
            with ui.row():
                ui.button('Сохранить состояние',icon='bookmark',on_click=lambda:simple('snapshot')).props('outline')
                ui.button('Вернуться к состоянию',icon='restore',on_click=lambda:simple('restore')).props('outline')
                ui.button('Проверить очередь · без движения',on_click=lambda:simple('allocation_check')).props('flat')
            allocation=ui.label('Проверка очереди использует два заданных запроса; это не обученная оценка риска.').classes('muted')
        async def tick():
            nonlocal busy,latest,schema,active
            if not started or busy:return
            busy=True
            try:
                if active:
                    if not await call('target',target=[float(x.value) for x in sliders]):active=False
                latest=await call()
                if latest:
                    if latest['owner']==operator:await call('heartbeat')
                    if latest['rgb']:camera.set_source('data:image/jpeg;base64,'+latest['rgb'])
                    key=(robot.value,latest['world']['robots'][robot.value]['task_id'])
                    if key!=schema:schema=key;controls.refresh()
                    result.set_text(' · '.join(NAMES[k]+': '+('успех' if v['outcome']=='success' else v['outcome'] or 'в процессе') for k,v in latest['world']['robots'].items()))
                    status.set_text(latest['error'] or ('Пауза' if latest['paused'] else 'Сцена работает')+f" · {latest['world']['time_s']:.2f} с")
                    if latest.get('allocation'):allocation.set_text('Два запроса → один оператор: '+NAMES.get(latest['allocation']['assignment'],'никто'))
            finally:busy=False
        ui.timer(.2,tick);ui.on('fleet_hidden',release);client.on_disconnect(release)
        ui.add_body_html("<script>document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('fleet_hidden')});window.addEventListener('pagehide',()=>emitEvent('fleet_hidden'));</script>")
