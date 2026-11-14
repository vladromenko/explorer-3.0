"""Operator pages; refresh only reads diagnostics, no automatic hardware acquisition."""
import asyncio
import json
from nicegui import ui
from . import fleet_config
from .paths import DATA
from .recording_readiness import Service,NAMES

STYLE='''
.nicegui-content{max-width:1280px;width:100%;margin:auto;padding:24px;box-sizing:border-box}
.ready-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:18px;width:100%}
.ready-card{background:#192331;border:1px solid #344253;border-radius:14px;padding:20px;min-width:0;width:100%;box-sizing:border-box}
.ready-item{border-bottom:1px solid #344253;padding:10px 0;width:100%;overflow-wrap:anywhere}
.ready-red{color:#ffb5a7}.ready-green{color:#7dd3c7}.ready-action{color:#ffe0a2;font-size:14px}
'''


def install(service=None):
    service=service or Service()

    @ui.page('/recording/readiness')
    async def readiness_page():
        ui.add_css(STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        with ui.row():
            ui.link('Explorer','/');ui.link('Сбор датасета','/recording/collect');ui.link('Инструкция','/datasets/guide')
        ui.label('Готовность парка').style('font-size:30px;font-weight:650')
        ui.label('Проверка связи и данных каждые 2 секунды. Открытие страницы не включает роботов.').style('color:#a6b6c6')
        stamp=ui.label('Собираем состояние…')
        general=ui.column().classes('ready-card')
        panels={}
        with ui.element('div').classes('ready-grid'):
            for robot,name in NAMES.items():
                with ui.column().classes('ready-card'):
                    ui.label(name).style('font-size:23px;font-weight:650')
                    async def toggle(event,r=robot):
                        try:
                            await asyncio.to_thread(fleet_config.update_robot,r,enabled=bool(event.value),root=DATA)
                            service.at=0
                        except Exception as exc:ui.notify(str(exc),type='negative')
                    try:enabled=fleet_config.load(DATA)['robots'][robot]['enabled']
                    except ValueError:enabled=False
                    ui.switch('Разрешить '+name,value=enabled,on_change=toggle)
                    panels[robot]=ui.column().classes('w-full')
                    with ui.row():
                        ui.link('Включить управление','/rosmaster/control' if robot=='rosmaster' else '/?robot='+robot)
                        ui.link('Проверка «всё ездит»','/recording/check/'+robot)
        busy=False
        def show(container,checks):
            container.clear()
            with container:
                for check in checks:
                    with ui.column().classes('ready-item'):
                        ui.label(('✓ ' if check['status']=='green' else '• ')+check['label']).classes('ready-'+check['status'])
                        value=check['value']
                        ui.label(json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value)).style('font-size:13px;color:#b9c9d8')
                        if check['action']:ui.label(check['action']).classes('ready-action')
        async def refresh():
            nonlocal busy
            if busy:return
            busy=True
            try:
                report=await asyncio.to_thread(service.snapshot)
                show(general,report['general'])
                for robot in NAMES:show(panels[robot],report['robots'][robot]['checks'])
                stamp.set_text('Обновлено · физическая приёмка выполняется отдельно')
            except Exception as exc:stamp.set_text('Не удалось обновить: '+str(exc))
            finally:busy=False
        ui.timer(2,refresh);await refresh()
    return service
