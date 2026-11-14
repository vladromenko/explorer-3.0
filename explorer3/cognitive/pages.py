"""Team workflow inside the existing Explorer UI, using the existing Jobs queue."""
import asyncio
import base64
import json
from nicegui import ui
from explorer3 import catalog
from explorer3.paths import DATA
from explorer3.sim_pages import STYLE
from . import store,broker,registry
from .run import ROLES

LABELS={'queued':'В очереди','running':'Выполняется в симуляции','completed':'Проверка завершена',
    'failed':'Остановлено с ошибкой','cancelled':'Отменено','interrupted':'Прервано перезапуском'}
OUTCOMES={'success':'подтверждён','failure':'не достигнут','unknown':'неизвестен','cancelled':'отменён'}


def history():
    return [j for j in catalog.rows('jobs') if json.loads(j['config']).get('kind')=='cognitive_simulation']


def cancel_job(jobs,ident,reason):
    if ident:
        path=DATA/'jobs'/ident/'mission.json'
        if path.is_file():broker.cancel(json.loads(path.read_text())['mission_id'],reason)
        jobs.cancel(ident)


def install(css,jobs):
    store.initialize();store.recover_interrupted()
    install_archive_review(css)
    @ui.page('/team')
    async def page():
        ui.add_css(css+STYLE+'''.team-grid{display:grid;grid-template-columns:1.5fr 1fr;gap:20px;width:100%}
        .team-cameras{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%}
        .team-tag{padding:5px 10px;border-radius:20px;background:#223d3d;color:#a5e7dc}
        @media(max-width:850px){.team-grid,.team-cameras{grid-template-columns:1fr}}''')
        ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        active=None;selected=None;last_heartbeat=0.;displayed=None;busy=False
        async def stop(reason='Остановлено оператором'):
            nonlocal active
            if active:
                ident=active;active=None
                await asyncio.to_thread(cancel_job,jobs,ident,reason)
                status.set_text('Отменено. Самостоятельного возобновления нет.')
        async def launch():
            nonlocal active,selected,last_heartbeat,displayed
            try:
                active=await asyncio.to_thread(jobs.cognitive_simulation,comparison.value,int(seed.value),scene.value)
                selected=active;last_heartbeat=__import__('time').monotonic();displayed=None
                status.set_text('В очереди · только симуляция');start.disable()
                saved.set_options({j['id']:j['id'][:8]+' · '+LABELS[j['status']] for j in history()},value=selected)
            except Exception as exc:status.set_text(str(exc))
        def heartbeat():
            nonlocal last_heartbeat
            last_heartbeat=__import__('time').monotonic()
        def choose(event):
            nonlocal selected,displayed
            if not active:selected=event.value;displayed=None
        ui.on('team_heartbeat',heartbeat)
        ui.on('team_hidden',lambda:stop('Вкладка скрыта; симуляционное задание отменено'))
        ui.context.client.on_disconnect(lambda:stop('Соединение с браузером потеряно'))
        ui.add_body_html('''<script>setInterval(()=>{if(!document.hidden)emitEvent('team_heartbeat')},500);
        document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('team_hidden')});
        window.addEventListener('pagehide',()=>emitEvent('team_hidden'));</script>''')
        with ui.row().classes('hero'):
            with ui.column().style('gap:3px'):
                ui.label('HOMEBRAIN / ОБЩАЯ ЗАДАЧА').classes('kicker')
                ui.html('<h1>Команда</h1>')
                ui.label('Наблюдение → помощь → действие → проверенный опыт').classes('muted')
            ui.button('STOP · задача',icon='stop_circle',color='negative',on_click=lambda:stop())
        with ui.row():
            ui.link('Explorer','/');ui.link('Общий стенд','/sim/fleet');ui.link('Реальные датчики','/fleet/live')
            ui.link('Наблюдения ROSMASTER','/team/rosmaster')
        ui.label('Симуляция · сценарные навыки · веса не изменяются').classes('team-tag')
        ui.label('SO-101 на столе; ROSMASTER и Tank на полу. Высота стола пока не измерена; общая калибровка не подтверждена.').classes('muted')
        with ui.tabs() as tabs:
            team=ui.tab('Команда');agents=ui.tab('Агенты');memory_tab=ui.tab('Общая память');results_tab=ui.tab('Помощь и результаты')
        with ui.tab_panels(tabs,value=team).classes('w-full'):
            with ui.tab_panel(team):
                with ui.element('div').classes('team-grid'):
                    with ui.column().classes('sim-panel'):
                        ui.label('Увидеть цель и подъехать').classes('sim-title')
                        ui.label('Tank не видит фиксированную цель. ROSMASTER направляет на неё свою камеру. Только новое подтверждённое наблюдение разрешает Tank подъехать и остановиться.').classes('muted')
                        ui.label('Общий мир · обзор, не вход политики').classes('muted')
                        overview=ui.image().props('no-spinner')
                        with ui.element('div').classes('team-cameras'):
                            with ui.column().classes('w-full'):
                                ui.label('Камера помощника · ROSMASTER');donor=ui.image().props('no-spinner')
                            with ui.column().classes('w-full'):
                                ui.label('Камера получателя · Tank');recipient=ui.image().props('no-spinner')
                        status=ui.label('Готово к проверке').classes('sim-status')
                        detail=ui.label('Модель предлагает план; отдельная проверка решает, можно ли его выполнить.').classes('muted')
                    with ui.column().classes('sim-panel'):
                        ui.label('Проверить взаимодействие').classes('sim-title')
                        ui.label('Первый запуск собирает опыт. Повтор использует подходящий случай, но снова проверяет цель и остановку.').classes('muted')
                        from explorer3.sim.room import PRESETS
                        scene=ui.select({k:v['label'] for k,v in PRESETS.items()},value='workshop',label='Окружение').classes('w-full')
                        seed=ui.number('Расстановка',value=17,min=0,max=2**31-1,precision=0)
                        comparison=ui.select({'M3':'С моделью и проверяемой памятью','M1':'С моделью, без памяти',
                            'M2':'Память без фильтра получателя · сравнение','M0':'Фиксированный план · без модели'},value='M3',label='Режим проверки').classes('w-full')
                        start=ui.button('Выполнить в симуляции',icon='play_arrow',on_click=launch)
                        ui.label('Реальные роботы не подключаются. Обучение не запускается. При уходе со страницы текущая задача отменяется.').classes('muted')
                        saved=ui.select({j['id']:j['id'][:8]+' · '+LABELS[j['status']] for j in history()},label='Сохранённые проверки',on_change=choose).classes('w-full')
                        links=ui.column()
            with ui.tab_panel(agents):
                ui.label('Шесть ролей · одна локальная модель').classes('sim-title')
                for name,purpose in ROLES.items():
                    with ui.card().classes('workspace'):
                        ui.label(name);ui.label(purpose).classes('muted')
                ui.label('Модель не получает SSH, моторные команды, Python, SQL или запуск обучения. G показывается как недоступный, пока нет пригодного обученного оценщика.').classes('muted')
                with ui.expansion('Доступные навыки').classes('w-full'):
                    for card in registry.cards().values():ui.label(card.skill_id+' · симуляция: '+card.sim_admission+' · реальные движения: '+card.real_admission)
            with ui.tab_panel(memory_tab):
                ui.label('Опыт с областью применимости').classes('sim-title')
                ui.label('Один успех остаётся кандидатом. Две независимые проверки поддерживают случай только для той же модели, камеры и навыка. Ошибки сохраняются.').classes('muted')
                memory_panel=ui.column().classes('w-full')
            with ui.tab_panel(results_tab):
                ui.label('Результат помощи и продолжения').classes('sim-title')
                outcomes=ui.column().classes('w-full');journal=ui.log(max_lines=100).classes('w-full').style('height:350px')
        last_memory=None;last_event=0
        async def refresh():
            nonlocal busy,active,displayed,last_memory,last_event
            if busy:return
            busy=True
            try:
                if active and __import__('time').monotonic()-last_heartbeat>5:await stop('Истёк пульс браузера')
                cases=await asyncio.to_thread(store.cases)
                memory_hash=store.digest(cases)
                if memory_hash!=last_memory:
                    last_memory=memory_hash;memory_panel.clear()
                    with memory_panel:
                        if not cases:ui.label('Проверенных случаев пока нет. Первый запуск сохранит исходные наблюдения и результат.').classes('muted')
                        for case in cases[:20]:
                            with ui.card().classes('workspace'):
                                ui.label(case['donor'].upper()+' → '+case['recipient'].upper()+' · '+case['status'])
                                ui.label(case['reason']).classes('muted')
                                ui.label('Помощник: '+OUTCOMES[case['helper_outcome']]+' · Получатель: '+OUTCOMES[case['recipient_outcome']]+' · Задача: '+OUTCOMES[case['task_outcome']])
                if selected:
                    job=await asyncio.to_thread(jobs.get,selected);work=DATA/'jobs'/selected
                    status.set_text(LABELS[job['status']]+' · '+selected[:8])
                    path=work/'mission.json'
                    if path.is_file():
                        mission_id=json.loads(path.read_text())['mission_id'];events=await asyncio.to_thread(store.events,mission_id)
                        if selected!=displayed:last_event=0;journal.clear()
                        for event in events:
                            if event['id']>last_event:
                                journal.push(event['kind']+' · '+json.dumps(event['body'],ensure_ascii=False)[:750]);last_event=event['id']
                        if events:detail.set_text(events[-1]['kind'])
                    for element,filenames in ((overview,['world-final.png','world-observation.png','world-before.png']),(donor,['donor-final.png','donor-observation.png','donor-before.png']),
                                              (recipient,['recipient-final.png','recipient-0000.png','recipient-before.png'])):
                        picture=next((work/'scene'/name for name in filenames if (work/'scene'/name).exists()),None)
                        if picture:element.set_source('data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode())
                    if displayed!=selected:
                        displayed=selected;links.clear()
                        with links:ui.link('Файлы и журнал проверки','/jobs/'+selected)
                    if job['status'] not in ('queued','running'):
                        if active==selected:active=None
                        start.enable();result_path=work/'result.json'
                        if result_path.is_file():
                            result=json.loads(result_path.read_text());outcomes.clear()
                            with outcomes:
                                for key,label in [('helper','Помощник'),('recipient','Получатель'),('task','Общая задача')]:
                                    ui.label(label+': '+OUTCOMES.get(result.get(key),'неизвестен'))
                                measured=result.get('proposal',{})
                                ui.label('План: '+measured.get('proposal',{}).get('choice','—'))
                                ui.label('Извлечено случаев: '+str(len(result.get('memory',{}).get('accepted',[]))))
                                if 'total_s' in measured:
                                    ui.label(f"Локальная модель: {measured['total_s']:.1f} с · пик RAM {measured['rss_peak_kb']/1024**2:.2f} ГиБ · offline")
                                ui.label('Это проверка оркестрации. Научная гипотеза и обученные моторные навыки здесь не проверяются.').classes('muted')
                        elif job.get('error'):detail.set_text(job['error'])
            except Exception as exc:detail.set_text(str(exc))
            finally:busy=False
        ui.timer(1,refresh)
        await refresh()


def install_archive_review(css):
    @ui.page('/team/rosmaster')
    async def archive_page():
        from . import shadow
        ui.add_css(css+STYLE);ui.dark_mode().enable()
        ui.colors(primary='#2b9c8f',negative='#d45050')
        ui.link('Команда','/team')
        with ui.column().classes('hero'):
            ui.label('ROSMASTER / ПОДГОТОВКА').classes('kicker')
            ui.html('<h1>Сохранённые наблюдения</h1>')
            ui.label('Разбор записи на Homebrain. Подключение к роботу и модель не используются.').classes('muted')
        with catalog.connect() as db:
            rows=db.execute("""SELECT body FROM cognitive_evidence
                WHERE json_extract(body,'$.robot')='rosmaster'
                AND json_extract(body,'$.domain')='real'
                AND json_extract(body,'$.source')='recorded_sensor_snapshot'
                ORDER BY expires DESC LIMIT 100""").fetchall()
        items=[json.loads(row['body']) for row in rows]
        if not items:
            ui.label('В памяти команды ещё нет импортированных записей ROSMASTER.').classes('sim-status')
            ui.label('Архив датчиков остаётся в прежнем разделе; наличие записи само по себе не подтверждает готовность робота.').classes('muted')
            ui.link('Архив и датчики ROSMASTER','/rosmaster/sensors')
        else:
            selection=ui.select({item['id']:item['id'][:8]+' · '+item['sample_id'] for item in items},
                value=items[0]['id'],label='Исходное наблюдение').classes('w-full')
            status=ui.label('Выберите запись для разбора.').classes('sim-status')
            panel=ui.column().classes('w-full')
            async def inspect():
                try:
                    result=await asyncio.to_thread(shadow.review,selection.value)
                    panel.clear()
                    labels={'missing':'Не записан','unknown':'Давность неизвестна',
                        'reported_recent_at_capture':'При записи сообщался свежим',
                        'reported_stale_at_capture':'Уже при записи был устаревшим'}
                    with panel:
                        for stream in result['streams']:
                            with ui.card().classes('workspace w-full'):
                                ui.label(stream['label']);ui.label(labels[stream['capture_status']]).classes('muted')
                        ui.label('Сейчас состояние робота неизвестно. Результат задачи не оценивался.').classes('sim-status')
                        ui.label('Исходные часы и контрольные суммы сохранены. Углы сервоприводов не заменяют калибровку, а локальные датчики — общую карту.').classes('muted')
                        ui.button('Скачать разбор',icon='download',on_click=lambda:ui.download(
                            store.encoded(result).encode(),'rosmaster-observation-review.json'))
                    status.set_text('Архив разобран · команд роботу не выдавалось')
                except Exception as exc:
                    panel.clear();status.set_text('Не удалось проверить исходную запись: '+str(exc))
            ui.button('Разобрать запись',icon='fact_check',on_click=inspect)
