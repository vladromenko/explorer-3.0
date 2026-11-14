"""Explicit read-only access to the ROSMASTER agent; no command API."""
import json
import time
import httpx
from fastapi import HTTPException
from fastapi.responses import Response
from nicegui import app, ui
from .paths import DATA


async def fetch(path):
    try:
        from .local_work import require_hardware_mode
        require_hardware_mode('rosmaster')
        cfg = json.loads((DATA/'configs/rosmaster-agent.json').read_text())
        async with httpx.AsyncClient(base_url=f"http://{cfg['host']}:{cfg['port']}",
                headers={'X-Explorer-Token':cfg['token']}, trust_env=False, timeout=2) as client:
            response = await client.get(path)
            response.raise_for_status()
            return response
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(503, 'ROSMASTER недоступен для наблюдения') from exc


@app.get('/api/rosmaster/observation')
async def observation():
    return (await fetch('/health')).json()


@app.get('/api/rosmaster/frame.jpg')
async def frame():
    result = await fetch('/frame.jpg')
    return Response(result.content, media_type='image/jpeg', headers={'Cache-Control':'no-store'})


@ui.page('/rosmaster')
def page():
    ui.dark_mode().enable()
    ui.link('← Explorer 3.0', '/')
    ui.label('ROSMASTER · наблюдение').style('font-size:26px')
    ui.label('Камера и измерения. Эта страница не включает приводы и не отправляет команды управления.')
    ui.label('Диагностический кадр обновляется раз в 2 секунды.')
    status = ui.label('Ожидание данных')
    image = ui.image().style('width:640px;max-width:100%')
    report = ui.code('',language='json').style('width:100%;max-height:450px;overflow:auto')
    async def refresh():
        try:
            result = await observation()
            state = result.get('observation') or {}
            outside = state.get('arm_outside_native_model') or []
            status.set_text(('Управление активно' if result.get('armed') else 'Связь есть · управление не включено') +
                (' · за пределами модели: '+', '.join(map(str,outside)) if outside else ''))
            report.set_content(json.dumps(state, ensure_ascii=False, indent=2))
            image.set_source('/api/rosmaster/frame.jpg?t='+str(time.monotonic_ns()))
        except HTTPException:
            status.set_text('ROSMASTER недоступен · прежние данные не являются текущими')
            image.set_source('')
            report.set_content('{}')
    ui.button('Обновить',on_click=refresh)
    ui.timer(2,refresh)
