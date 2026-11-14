"""Operator pages; refresh only reads diagnostics, no automatic hardware acquisition."""
import asyncio
import json
from nicegui import ui
from . import fleet_config
from .paths import DATA
from .recording_readiness import Service,NAMES

STYLE='''
.nicegui-content{max-width:1280px;width:100%;margin:auto;padding:24px;box-sizing:border-box}
.w-full{width:100%}.ready-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:18px;width:100%}
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


def install_collection():
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from nicegui import app
    from urllib.parse import urlencode
    from . import collection,catalog

    @app.get('/recording/media/{episode}/{kind}')
    def media(episode:str,kind:str):
        import re
        if not re.fullmatch('[a-f0-9]{32}',episode):raise HTTPException(404)
        if kind in {'first','last'}:path=DATA/'validation/collection'/episode/(kind+'.jpg')
        elif kind=='video':
            entry=next((r for r in catalog.rows('episodes') if r['id']==episode),None)
            if entry is None:raise HTTPException(404)
            root=__import__('pathlib').Path(entry['dataset']).resolve()
            if not root.is_relative_to(DATA/'datasets'):raise HTTPException(404)
            path=next((root/'videos').rglob('*.mp4'),None)
        else:raise HTTPException(404)
        if path is None or not path.is_file():raise HTTPException(404)
        return FileResponse(path)

    @ui.page('/recording/collect')
    async def collection_page():
        ui.add_css(STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        with ui.row():
            ui.link('Explorer','/');ui.link('Готовность парка','/recording/readiness');ui.link('Инструкция','/datasets/guide')
        ui.label('Сбор датасета').style('font-size:30px;font-weight:650')
        ui.label('10 расстановок × 5 успешных показов на задачу. Повтор прежней расстановки остаётся в той же группе.').style('color:#a6b6c6')
        selected=[None];last_report=[None];loading=[False]
        with ui.column().classes('ready-card'):
            task=ui.select({k:v[0] for k,v in collection.TASKS.items()},value='tank_dock',label='Задача').classes('w-full')
            variant=ui.select({},label='Вариант потоков').classes('w-full')
            operator=ui.input('Оператор',value='operator').classes('w-full')
            description=ui.input('Что изменилось в расстановке').classes('w-full')
            repeat=ui.select({'':'Новая расстановка'},value='',label='Расстановка').classes('w-full')
            choice=ui.select({},label='Продолжить серию').classes('w-full')
            notice=ui.label('Выберите серию или создайте новую')
            async def create():
                try:
                    record=await asyncio.to_thread(collection.create_series,task.value,variant.value,operator.value,description.value,repeat.value or None)
                    setup();choice.set_value(record['id']);open_series()
                except Exception as exc:ui.notify(str(exc),type='negative')
            new_button=ui.button('Новая серия',on_click=create)
        counters=ui.label('').style('font-size:18px')
        progress=ui.label('')
        ui.label('В панели ниже: «Начать показ», затем «Успех», «Неуспех» или «Прервать». F9 — начать, F10 — успех, F11 — неуспех; пробел — STOP. Клавиши записи работают при фокусе в панели.').style('color:#a6b6c6')
        frame=ui.element('iframe').props('id="collection-control" src="about:blank" allow="gamepad"').style('width:100%;height:1250px;border:1px solid #344253;border-radius:14px;background:#10161f')
        ui.add_body_html('''<script>window.addEventListener('keydown',e=>{
          if(e.code==='Space'&&!['INPUT','TEXTAREA'].includes(e.target.tagName)){
            e.preventDefault();document.getElementById('collection-control')?.contentWindow.postMessage({explorerRecordingStop:true},location.origin);
          }
        });</script>''')
        review=ui.column().classes('ready-card')
        review_key=[None]
        def setup():
            existing=collection.series(task.value)
            variants=collection.TASKS[task.value][1]
            chosen=existing[0]['profile'] if existing else variants[0]
            variant.set_options({v:v for v in variants},value=chosen)
            variant.set_enabled(not existing)
            repeat.set_options({'':'Новая расстановка',**{s['layout']:s['layout']+' · общая группа' for s in existing}},value='')
            choice.set_options({s['id']:s['id']+' · '+s['description'] for s in existing},value=None)
        def open_series():
            if not choice.value:return
            record=next(s for s in collection.series(task.value) if s['id']==choice.value)
            selected[0]=record['id'];review_key[0]=None
            robot=record['task'].split('_')[0]
            url=('/rosmaster/control' if robot=='rosmaster' else '/')+'?'+urlencode(
                {'robot':robot,'series':record['id'],'profile':record['profile'],'layout':record['layout']})
            frame.props('src='+json.dumps(url))
            notice.set_text(record['id']+' · '+record['layout']+' · '+record['description'])
        def task_changed():
            selected[0]=None;frame.props('src="about:blank"');setup()
        task.on_value_change(task_changed);choice.on_value_change(open_series);setup()
        async def refresh():
            if loading[0] or not selected[0]:return
            loading[0]=True
            try:
                report=await asyncio.to_thread(collection.summary,selected[0]);last_report[0]=report
                c=report['current'];t=report['total']
                counters.set_text(f"Серия: {c['success']} успех / {c['failure']} неуспех / {c['aborted']} прервано · всего: {t['success']} / {t['failure']} / {t['aborted']} · до 50: {report['to_50']} · групп с проверенными показами: {report['groups_with_accepted_data']} из {report['independent_groups']} расстановок")
                active=any(a['phase'] in {'preparing','recording'} for a in report['attempts'])
                for element in (task,choice,new_button):element.set_enabled(not active)
                latest=report['attempts'][-1] if report['attempts'] else None
                estimate=report['estimated_remaining_bytes']
                space='оценка появится после первого показа' if estimate is None else f"на оставшиеся: {estimate/1024**3:.2f} ГБ"
                progress.set_text((latest['phase']+' · '+latest['streams'] if latest else 'Записей пока нет')+
                    f" · средняя длительность: {report['mean_duration_s']} с · {space}")
                key=tuple((a['id'],a['phase'],a['quality']) for a in report['attempts'])
                if key!=review_key[0]:
                    review_key[0]=key;review.clear()
                    with review:
                        ui.label('Посмотрите 1–2 записи этой серии').style('font-size:20px')
                        for attempt in reversed(report['attempts']):
                            if attempt['phase']=='saved':
                                quality=json.loads(attempt['quality'] or '{}')
                                ui.label(attempt['outcome']+' · '+str(attempt['frames'])+' кадров')
                                if quality.get('status')!='pass':
                                    ui.label(' · '.join(quality.get('problems',[]))).classes('ready-action')
                                if attempt['episode']:
                                    episode=attempt['episode']
                                    ui.button('Исключить из обучения',on_click=lambda e=episode:collection.exclude(e))
                                    with ui.row():
                                        for kind in quality.get('thumbnails',[]):ui.image(f'/recording/media/{episode}/{kind}').style('width:150px;max-width:100%')
                                    with ui.expansion('Посмотреть запись').classes('w-full'):
                                        ui.video(f'/recording/media/{episode}/video').props('preload="none"').style('width:100%;max-width:640px')
            except Exception as exc:progress.set_text('Не удалось обновить серию: '+str(exc))
            finally:loading[0]=False
        ui.timer(1,refresh)
