"""User-reviewed image perception, separate from policy actions and ground truth."""
import asyncio
import base64
import json
from pathlib import Path
from nicegui import ui
from . import sim_service
from .fleet_service import service as fleet
from .sim_pages import STYLE
from .paths import DATA


def install(css,jobs):
    @ui.page('/sim/vision')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        raw=None;source=None;ident=None;busy=False;displayed=None
        async def capture(shared=False):
            nonlocal raw,source
            try:
                state=await asyncio.to_thread(fleet.request if shared else sim_service.request)
                if not state.get('rgb'):raise ValueError('Сначала откройте нужную сцену')
                raw=state['rgb'];source='fleet_sim' if shared else 'individual_sim'
                original.set_source('data:image/jpeg;base64,'+raw);status.set_text('Кадр сохранён для анализа')
            except Exception as exc:status.set_text(str(exc))
        async def uploaded(event):
            nonlocal raw,source
            content=await event.file.read()
            if len(content)>8_000_000:status.set_text('Файл больше 8 МБ');return
            raw=base64.b64encode(content).decode();source='uploaded'
            original.set_source('data:image/png;base64,'+raw);status.set_text('Изображение загружено')
        async def launch():
            nonlocal ident
            try:
                if raw is None:raise ValueError('Выберите кадр сцены или загрузите изображение')
                ident=await asyncio.to_thread(jobs.inspect_image,raw,method.value,prompt.value,source)
                status.set_text('Анализ в очереди · '+ident[:8]);overlay.set_source('');summary.set_text('')
            except Exception as exc:status.set_text(str(exc))
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / ЗРЕНИЕ').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Найти и выделить объект по описанию').classes('muted')
            ui.label('Без обучения и движения').classes('sim-badge')
        with ui.row():
            ui.link('← Симуляция','/sim');ui.link('Общий стенд','/sim/fleet')
        with ui.element('div').classes('sim-layout'):
            with ui.column().classes('sim-panel'):
                ui.label('Исходное изображение').classes('sim-title');original=ui.image().props('no-spinner')
                with ui.row():
                    ui.button('Кадр отдельной сцены',icon='photo_camera',on_click=lambda:capture(False))
                    ui.button('Кадр общего стенда',on_click=lambda:capture(True)).props('outline')
                ui.upload(label='Или загрузить изображение',auto_upload=True,on_upload=uploaded,max_file_size=8_000_000).props('accept=image/*').classes('w-full')
            with ui.column().classes('sim-panel'):
                ui.label('Что найти').classes('sim-title')
                prompt=ui.input('Короткое описание объекта',value='orange cube').classes('w-full')
                ui.label('Например: orange cube, green box. Названия на английском подходят обеим моделям.').classes('muted')
                method=ui.select({'yoloe':'YOLOE · быстрый поиск и маски','sam3':'SAM 3 · выделение по описанию'},value='yoloe',label='Модель зрения').classes('w-full')
                ui.button('Проанализировать кадр',icon='image_search',on_click=launch)
                status=ui.label('Выберите изображение').classes('sim-status')
                ui.label('Работает локально на Homebrain. GPU-анализ ждёт свободных ресурсов. Результаты не отправляются роботам и не добавляются в обучающие данные автоматически.').classes('muted')
        with ui.column().classes('sim-panel'):
            ui.label('Результат для проверки').classes('sim-title');overlay=ui.image().props('no-spinner').style('max-width:960px')
            summary=ui.label('').classes('sim-status')
            ui.button('Скачать отчёт',icon='download',on_click=lambda:ui.download.file(DATA/'jobs'/ident/'result.json') if ident and (DATA/'jobs'/ident/'result.json').exists() else None).props('outline')
            ui.label('Модель может пропустить объект или ошибиться. Маски — предложения для проверки; они не подтверждают успех задачи.').classes('muted')
        async def poll():
            nonlocal busy,displayed
            if ident is None or busy:return
            busy=True
            try:
                job=jobs.get(ident);status.set_text({'queued':'Ожидает свободных ресурсов','running':'Анализ изображения','completed':'Анализ завершён','failed':'Ошибка анализа','cancelled':'Анализ отменён'}.get(job['status'],job['status']))
                if job['status']=='completed' and displayed!=ident:
                    result=json.loads((DATA/'jobs'/ident/'result.json').read_text())
                    overlay.set_source(Path(result['overlay']));displayed=ident
                    summary.set_text(f"Найдено объектов: {len(result['scores'])} · расчёт {result['inference_seconds']:.2f} с · загрузка модели {result['load_seconds']:.2f} с")
                elif job['status']=='failed':summary.set_text(job['error'] or 'См. журнал задания '+ident)
            finally:busy=False
        ui.timer(1.,poll)
