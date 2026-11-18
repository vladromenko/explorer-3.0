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
        from . import data_safety
        with ui.expansion('Резервная копия').classes('ready-card'):
            ui.label('Без внешнего диска сохраняются SQLite и хеши, но не вторая копия самих видео. Автоматическая копия — ежедневно.')
            external=ui.input('Папка внешнего диска',placeholder='/media/…/explorer3-backup').classes('w-full')
            async def save_destination():
                try:
                    await asyncio.to_thread(data_safety.configure_external,external.value or '')
                    ui.notify('Путь сохранён')
                except Exception as exc:ui.notify(str(exc),type='negative')
            async def copy_now():
                copy_button.disable()
                try:
                    result=await asyncio.to_thread(data_safety.backup_now,force=True)
                    ui.notify(result.get('warning') or 'Копия создана, хеши проверены')
                    service.at=0
                except Exception as exc:ui.notify('Не удалось сохранить копию: '+str(exc),type='negative')
                finally:copy_button.enable()
            ui.button('Сохранить путь',on_click=save_destination)
            copy_button=ui.button('Сделать копию сейчас',on_click=copy_now)
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

    HOWTO={
        'rosmaster_pick_place':('Ведущая рука SO-101 (Leader) → рука ROSMASTER: «Включить зеркало» в пульте.',[
            'ROSMASTER на полу, колёсами не ездим. Предмет перед роботом в 16–26 см.',
            'Начальная поза: клешня над предметом ~10 см, открыта; предмет внизу по центру картинки.',
            'F9 → дождаться «Запись идёт» → опустить, сжать, поднять на 5+ см, держать 2 с → F10 (не взяли — F11).',
            'Примерно каждый пятый показ нарочно мимо на 2–3 см, затем поправиться и взять.',
            '5 успешных показов в расстановке → «Следующая расстановка»: другой предмет или место.']),
        'so101_pick_place':('Leader → Follower на столе; камера запястья и верхняя C920.',[
            'Стартовая поза одна и та же: рука сложена, захват открыт.',
            'F9 → взять предмет, поднять на 5+ см, удержать 2 с (или протянуть человеку для «передачи») → F10.',
            'Верхнюю камеру и основание руки не сдвигать.',
            '5 успешных показов в расстановке → «Следующая расстановка».']),
        'tank_dock':('Пульт Tank: клавиши или джойстик.',[
            'Tank в 1–2 м от цели, цель видна камере.',
            'F9 → подъехать к цели и остановиться в 20–30 см → F10.',
            '5 успешных показов в расстановке → «Следующая расстановка»: другое место старта или цель.']),
        'rosmaster_dock':('Пульт ROSMASTER: колёса W/A/S/D или джойстик.',[
            'ROSMASTER в 1–2 м от цели, цель видна камере.',
            'F9 → подъехать к цели и остановиться в 20–30 см → F10.',
            '5 успешных показов в расстановке → «Следующая расстановка».']),
    }
    EXAMPLES={'rosmaster_pick_place':'взять с пола: флакон, слева','so101_pick_place':'передача: кубик',
              'tank_dock':'подъехать: к коробке у двери','rosmaster_dock':'подъехать: к стулу'}
    TARGET=50;PER_LAYOUT=5
    COLLECT_CSS='''
.task-card{cursor:pointer;background:#192331;border:2px solid #344253;border-radius:14px;padding:14px 16px;min-width:220px;flex:1}
.task-card.on{border-color:#2b9c8f;background:#17302d}.task-card .t{font-size:16px;font-weight:600}.task-card .s{color:#a6b6c6;font-size:13px}
.pbar{height:8px;background:#26313f;border-radius:5px;overflow:hidden;width:100%}.pbar>i{display:block;height:100%;background:#2b9c8f}
.shot{background:#192331;border:1px solid #344253;border-radius:12px;padding:10px;width:230px}
.chip{padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}.c-ok{background:#1f3a2c;color:#7dd3c7}.c-bad{background:#3a1f1f;color:#ffb5a7}.c-mid{background:#33301f;color:#ffe0a2}
'''

    @ui.page('/recording/collect')
    async def collection_page(task:str='',series:str=''):
        ui.add_css(STYLE+COLLECT_CSS)
        ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        with ui.row().classes('items-center'):
            ui.link('Explorer','/');ui.link('Готовность парка','/recording/readiness');ui.link('Инструкция','/datasets/guide')
        ui.label('Сбор датасета').style('font-size:28px;font-weight:650')
        ui.label(f'Для каждой задачи: {TARGET//PER_LAYOUT} расстановок × {PER_LAYOUT} успешных показов = {TARGET}. '
                 'Неудачи не удаляются — это тоже данные.').style('color:#a6b6c6')
        overview={'v':await asyncio.to_thread(collection.overview)}
        first=task if task in collection.TASKS else max(overview['v'],key=lambda k:overview['v'][k]['last_used'])
        state={'task':first,'series':series or overview['v'][first]['latest']}
        cards=ui.row().classes('w-full').style('gap:12px')
        layout_card=ui.column().classes('ready-card').style('gap:8px')
        howto=ui.expansion('Как записывать',value=True).classes('ready-card')
        so101_box=ui.column().classes('w-full')
        panel_hint=ui.label('').style('color:#a6b6c6')
        frame=ui.element('iframe').props('id="collection-control" src="about:blank" allow="gamepad"').style(
            'width:100%;height:1150px;border:1px solid #344253;border-radius:14px;background:#10161f')
        ui.add_body_html('''<script>window.addEventListener('keydown',e=>{
          if(e.code==='Space'){
            if(!['INPUT','TEXTAREA'].includes(e.target.tagName))e.preventDefault();document.getElementById('collection-control')?.contentWindow.postMessage({explorerRecordingStop:true},location.origin);
          }
        });</script>''')
        ui.label('Показы этой расстановки').style('font-size:20px;margin-top:8px')
        shots=ui.row().classes('w-full').style('gap:12px')
        shots_key=[None];busy=[False]

        def draw_cards():
            cards.clear()
            with cards:
                for key,(label,_) in collection.TASKS.items():
                    o=overview['v'][key]
                    card=ui.element('div').classes('task-card'+(' on' if key==state['task'] else ''))
                    card.on('click',lambda k=key:select_task(k))
                    with card:
                        ui.label(label).classes('t')
                        ui.label(f"годных успехов {o['usable']} из {TARGET} · расстановок {o['layouts']}").classes('s')
                        ui.html(f'<div class="pbar"><i style="width:{min(100,100*o["usable"]/TARGET):.0f}%"></i></div>')

        dialog=ui.dialog()
        with dialog,ui.card().style('min-width:min(520px,92vw)'):
            ui.label('Следующая расстановка').style('font-size:20px')
            d_desc=ui.input('Что поставили (по-русски)').classes('w-full')
            d_instr=ui.input('Команда для робота (англ., подставится сама)').classes('w-full')
            d_desc.on_value_change(lambda _: d_instr.set_value(collection.suggest_instruction(state['task'],d_desc.value) or ''))
            with ui.expansion('Дополнительно').classes('w-full'):
                d_repeat=ui.select({'':'Нет — новая расстановка'},value='',label='Повтор прежней расстановки (та же группа)').classes('w-full')
                d_variant=ui.select({},label='Вариант потоков').classes('w-full')
                d_operator=ui.input('Оператор',value='operator').classes('w-full')
            async def create():
                try:
                    record=await asyncio.to_thread(collection.create_series,state['task'],d_variant.value,d_operator.value,
                                                   d_desc.value,d_repeat.value or None,d_instr.value)
                    dialog.close();await reload();open_series(record['id'])
                except Exception as exc:ui.notify(str(exc),type='negative')
            with ui.row():
                ui.button('Создать и открыть пульт',on_click=create).props('no-caps')
                ui.button('Отмена',on_click=dialog.close).props('flat no-caps')

        def open_dialog():
            existing=overview['v'][state['task']]['series']
            variants=collection.TASKS[state['task']][1]
            d_variant.set_options({v:v for v in variants},value=existing[0]['profile'] if existing else variants[0])
            d_variant.set_enabled(not existing)
            d_repeat.set_options({'':'Нет — новая расстановка',**{r['layout']:r['layout'] for r in existing}},value='')
            d_desc.set_value('');d_desc.props('placeholder="например: '+EXAMPLES.get(state['task'],'')+'"');d_instr.set_value('')
            dialog.open()

        def draw_layout(report=None):
            layout_card.clear()
            rows=overview['v'][state['task']]['series']
            with layout_card:
                if not rows:
                    ui.label('В этой задаче ещё нет расстановок.').style('font-size:16px')
                    ui.button('Создать первую расстановку',on_click=open_dialog).props('no-caps')
                    return
                current=next((r for r in rows if r['id']==state['series']),rows[-1])
                index=[r['id'] for r in rows].index(current['id'])+1
                c=(report or {}).get('current',{'success':0,'failure':0,'aborted':0})
                with ui.row().classes('items-center w-full').style('gap:14px'):
                    ui.label(f"Расстановка {index}: {current['description']}").style('font-size:18px;font-weight:600')
                    if current.get('instruction'):ui.label('«'+current['instruction']+'»').style('color:#a6b6c6')
                with ui.row().classes('items-center w-full').style('gap:14px'):
                    ui.label(f"в ней: успех {c['success']} из {PER_LAYOUT} · неуспех {c['failure']} · прервано {c['aborted']}")
                    if c['success']>=PER_LAYOUT:ui.label('Готово — переходите к следующей расстановке').classes('chip c-ok')
                with ui.row().classes('items-center').style('gap:10px'):
                    ui.button('Следующая расстановка',on_click=open_dialog).props('no-caps')
                    sel=ui.select({r['id']:r['id'].rsplit('-',1)[-1]+' · '+r['description'] for r in rows},value=current['id'],
                                  label='Открыть другую расстановку').style('min-width:300px')
                    sel.on_value_change(lambda e:open_series(e.value) if e.value!=state['series'] else None)

        def draw_howto():
            howto.clear()
            control,steps=HOWTO.get(state['task'],('',[]))
            with howto:
                ui.label(control).style('color:#a6b6c6')
                for i,step in enumerate(steps,1):ui.label(f'{i}. {step}')
                ui.label('Клавиши: F9 — начать, F10 — успех, F11 — неуспех, пробел — STOP. Клавиши работают, когда курсор в пульте.').style('color:#a6b6c6;font-size:13px')
            so101_box.clear()
            if state['task']=='so101_pick_place':
                with so101_box:so101_setup()

        def open_series(ident):
            rows=overview['v'][state['task']]['series']
            record=next((r for r in rows if r['id']==ident),None)
            shots_key[0]=None;shots.clear()
            if record is None:
                state['series']=None;frame.props('src="about:blank"');frame.style('display:none')
                panel_hint.set_text('Создайте расстановку — здесь появится пульт робота.');draw_layout();return
            state['series']=record['id']
            robot=record['task'].split('_')[0]
            url=('/rosmaster/control' if robot=='rosmaster' else '/')+'?'+urlencode(
                {'robot':robot,'series':record['id'],'profile':record['profile'],'layout':record['layout']})
            frame.props('src='+json.dumps(url));frame.style('display:block')
            panel_hint.set_text('Пульт робота ниже. Перед записью включите управление (для руки ROSMASTER — «Включить зеркало»), затем F9.')
            ui.run_javascript("history.replaceState(null,'','?'+"+json.dumps(urlencode({'task':state['task'],'series':record['id']}))+")")
            draw_layout()

        async def reload():
            overview['v']=await asyncio.to_thread(collection.overview);draw_cards()

        async def select_task(key):
            if key==state['task']:return
            state['task']=key;await reload();draw_howto();open_series(overview['v'][key]['latest'])

        async def refresh():
            if busy[0] or not state['series']:return
            busy[0]=True
            try:
                report=await asyncio.to_thread(collection.summary,state['series'])
                key=tuple((a['id'],a['phase'],a['quality'],a['user_excluded']) for a in report['attempts'])
                if key==shots_key[0]:return
                shots_key[0]=key;await reload();draw_layout(report);shots.clear()
                with shots:
                    if not report['attempts']:ui.label('Пока нет показов в этой расстановке.').style('color:#a6b6c6')
                    for a in reversed(report['attempts']):
                        q=json.loads(a['quality'] or '{}')
                        with ui.column().classes('shot').style('gap:4px'):
                            if a['phase']!='saved':
                                ui.label('идёт запись' if a['phase'] in ('preparing','recording') else 'проверяется…').classes('chip c-mid')
                                continue
                            name={'success':'Успех','failure':'Неуспех','aborted':'Прервано'}.get(a['outcome'],a['outcome'])
                            cls='c-ok' if a['outcome']=='success' and q.get('status')=='pass' else 'c-mid' if a['outcome']=='success' else 'c-bad'
                            with ui.row().classes('items-center').style('gap:6px'):
                                ui.label(name).classes('chip '+cls)
                                ui.label(f"{a['frames']} кадров"+(f" · {a['duration']:.0f} с" if a['duration'] else '')).style('font-size:12px;color:#a6b6c6')
                            if a['user_excluded']:ui.label('исключён из обучения').classes('chip c-bad')
                            elif q.get('status')!='pass':
                                ui.label('не пойдёт в обучение: '+'; '.join(q.get('problems',[])[:2])).style('font-size:12px;color:#ffe0a2')
                            e=a['episode']
                            if e and 'first' in q.get('thumbnails',[]):
                                ui.image(f'/recording/media/{e}/first').style('width:100%;border-radius:8px')
                            if e:
                                with ui.row().style('gap:4px'):
                                    def play(e=e):
                                        with ui.dialog() as d,ui.card():
                                            ui.video(f'/recording/media/{e}/video').style('width:min(640px,90vw)')
                                        d.open()
                                    ui.button('Видео',on_click=play).props('flat dense no-caps')
                                    if not a['user_excluded']:
                                        def drop(e=e):
                                            collection.exclude(e);shots_key[0]=None
                                        ui.button('Исключить',on_click=drop).props('flat dense no-caps color=negative')
            except Exception as exc:panel_hint.set_text('Не удалось обновить: '+str(exc))
            finally:busy[0]=False

        def so101_setup():
            from . import so101_geometry
            from .paths import ROOT
            cfg = so101_geometry.configuration()
            with ui.expansion('Опыт SO-101 для переноса захвата: верхняя камера и метка').classes('ready-card'):
                ui.label('Для донорских захватов: Leader/Follower, камера запястья и верхняя C920. Метка — по центру верхней грани кубика.')
                ui.button('Скачать метку ArUco №7 · 20 мм',on_click=lambda: ui.download.file(ROOT/'assets/so101_donor_marker_7.svg')).props('outline no-caps')
                marker_kind = ui.select({'DICT_APRILTAG_36h11:1': 'AprilTag 36h11 №1 (уже на кубике)', 'DICT_4X4_50:7': 'ArUco 4×4 №7 (распечатать)'},
                    value=f"{cfg.get('marker_dictionary', 'DICT_APRILTAG_36h11')}:{cfg.get('marker_id', 1)}", label='Метка на кубике').classes('w-full')
                geometry_enabled = ui.checkbox('Сохранять верхнюю камеру и геометрию в показах SO-101', value=cfg.get('enabled', False))
                instance = ui.input('Экземпляр кубика', value=cfg.get('instance_id', '')).classes('w-full')
                dimensions = cfg.get('size_m', [.032] * 3)
                sizes = [ui.number(label, value=dimensions[i]*1000, min=10, max=100).classes('w-full')
                         for i, label in enumerate(('Ширина, мм', 'Длина, мм', 'Высота, мм'))]
                marker_size = ui.number('Сторона чёрной метки, мм', value=cfg.get('marker_side_m', .020)*1000, min=5, max=100).classes('w-full')
                def save_geometry():
                    try:
                        so101_geometry.configure_capture(enabled=bool(geometry_enabled.value), instance_id=instance.value,
                            marker_side_m=float(marker_size.value)/1000, size_m=[float(v.value)/1000 for v in sizes],
                            marker_dictionary=marker_kind.value.split(':')[0], marker_id=int(marker_kind.value.split(':')[1]))
                        ui.notify('Сохранено. Робот не включён.')
                    except Exception as exc:
                        ui.notify(str(exc), type='negative')
                ui.button('Сохранить', on_click=save_geometry).props('no-caps')

        draw_cards();draw_howto();open_series(state['series'])
        ui.timer(1,refresh)


def install_physical(local_reader=lambda:{}):
    from .physical_acceptance import Run,Reader,STEPS
    from .paths import ROOT

    @ui.page('/recording/check/{robot}')
    async def physical_page(robot:str):
        if robot not in STEPS:
            ui.label('Выберите SO-101, Tank или ROSMASTER');return
        ui.add_css(STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        with ui.row():
            ui.link('Готовность парка','/recording/readiness');ui.link('Сбор датасета','/recording/collect',new_tab=True)
        ui.label('Проверка «всё ездит» · '+NAMES[robot]).style('font-size:28px;font-weight:650')
        ui.label(('Только вперёд и назад. ' if robot=='tank' else 'Около 10 минут. ')+'Мастер только читает датчики. Управляете вы в панели ниже; оставайтесь рядом с выключателем.').classes('ready-action')
        if robot=='tank':
            ui.label('Зарядник должен быть отсоединён. Напряжение само по себе не доказывает это — нужно ваше подтверждение.').classes('ready-action')
        run=Run(robot,lambda:Reader(robot,local_reader))
        steps=dict(STEPS[robot])
        selector=ui.select({k:str(i+1)+'. '+v for i,(k,v) in enumerate(STEPS[robot])},value=STEPS[robot][0][0],label='Шаг').classes('w-full')
        instruction=ui.label(steps[selector.value]).style('font-size:18px')
        confirmation=ui.checkbox('Я выполнил шаг и лично увидел указанный результат')
        status=ui.label('Измерение ещё не началось')
        result_box=ui.column().classes('ready-card')
        def selected():
            instruction.set_text(steps[selector.value]);confirmation.value=False
            confirmation.set_text('Зарядник физически отсоединён' if selector.value=='charger' else 'Я выполнил шаг и лично увидел указанный результат')
        selector.on_value_change(selected);selected()
        def start():
            try:
                run.start(selector.value);selector.disable();start_button.disable();confirmation.value=False
                status.set_text('Измерение идёт. Выполните шаг в панели, затем подтвердите результат.')
            except Exception as exc:ui.notify(str(exc),type='negative')
        async def finish():
            finish_button.disable()
            try:
                result=await asyncio.to_thread(run.finish,confirmation.value)
                result_box.clear()
                with result_box:
                    ui.label(result['result']).classes('ready-green' if result['result']=='PASS' else 'ready-red')
                    for check in result['checks']:
                        ui.label(('✓ ' if check['pass'] else '• ')+check['label']+((' · '+str(check['value'])) if 'value' in check else ''))
                        if check['action']:ui.label(check['action']).classes('ready-action')
                    ui.label('Сохранено: '+str(run.path.relative_to(DATA)))
                    ui.label('FAIL можно повторить. Автоматического продолжения управления нет.')
                selector.enable();start_button.enable()
                status.set_text('Шаг сохранён; выберите следующий или повторите этот')
            except Exception as exc:ui.notify(str(exc),type='negative')
            finally:finish_button.enable()
        with ui.row():
            start_button=ui.button('Начать измерение шага',on_click=start)
            finish_button=ui.button('Проверить и сохранить результат',on_click=finish)
        progress=ui.label('0 показаний')
        def refresh():
            with run.lock:samples=list(run.samples)
            progress.set_text(str(len(samples))+' показаний'+(' · '+samples[-1]['read_error'] if samples and samples[-1].get('read_error') else ''))
            if run.thread and not run.thread.is_alive() and run.current:status.set_text('Чтение закончено; подтвердите и сохраните результат или отметьте FAIL')
        ui.timer(.5,refresh)
        ui.label('Панель управления · движение только по вашим кнопкам').style('font-size:20px')
        src='/rosmaster/control' if robot=='rosmaster' else '/?robot='+robot
        control_frame=ui.element('iframe').props('id="physical-control" src='+json.dumps(src)+' allow="gamepad"').style('width:100%;height:1250px;border:1px solid #344253;border-radius:14px;background:#10161f')
        ui.add_body_html('''<script>window.addEventListener('keydown',e=>{
          if(e.code==='Space'){e.preventDefault();document.getElementById('physical-control')?.contentWindow.postMessage({explorerRecordingStop:true},location.origin);}
        });</script>''')
        ui.context.client.on_disconnect(lambda:run.done.set())
