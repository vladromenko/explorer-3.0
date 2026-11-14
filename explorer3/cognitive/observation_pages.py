"""Review recorded sensor evidence and prepare comparable experiments in Explorer."""
import asyncio
import base64
import json
from nicegui import ui
from explorer3 import catalog
from explorer3.paths import DATA
from explorer3.sensors import archive_entries,open_archive
from explorer3.sim_pages import STYLE
from . import evidence,observer,store,experiments


def install(css,jobs):
    @ui.page('/team/observations')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        ui.link('← Команда','/team')
        ui.html('<h1>Наблюдения команды</h1>')
        ui.label('Кадры и датчики из сохранённых записей. Движение не запускается.').classes('muted')
        archives=archive_entries('rosmaster')+archive_entries('tank')
        options={a['id']:a['robot'].upper()+' · датчики · '+a['id'][-8:] for a in archives}
        options.update({'episode:'+e['id']:e['profile']+' · показ · '+e['id'][:8] for e in catalog.rows('episodes')
            if not e['profile'].endswith('_sim') and e['frames']>0 and e['dataset']})
        selection=ui.select(options,value=next(iter(options),None),label='Запись или показ').classes('w-full')
        index=ui.number('Номер кадра',value=0,min=0,precision=0)
        status=ui.label('Выберите запись и откройте кадр.').classes('sim-status')
        image=ui.image().style('max-width:800px');details=ui.column().classes('w-full')
        frame_id=None;active=None;last_result=None;generation=0;busy=False
        async def open_frame():
            nonlocal frame_id,busy
            if busy or active:
                status.set_text('Сначала закончите или отмените текущий разбор.');return
            mine=generation;busy=True
            frame_id=None;model_button.disable();detector_button.disable();segment_button.disable();shadow_button.disable()
            try:
                n=int(index.value)
                streams=[]
                if selection.value and selection.value.startswith('episode:'):
                    frame=await asyncio.to_thread(observer.frame_from_episode,selection.value.split(':',1)[1],n)
                else:
                    root,rows=await asyncio.to_thread(open_archive,selection.value)
                    if not 0<=n<len(rows):raise ValueError('Номер вне записанной последовательности')
                    path=(root/rows[n]['file']).resolve()
                    if not path.is_relative_to(root.resolve()):raise ValueError('Неверный путь кадра')
                    item=await asyncio.to_thread(evidence.import_sensor_packet,str(path.relative_to(DATA)))
                    frame=await asyncio.to_thread(observer.frame_from_packet,item.id);streams=item.payload['stream_names']
                if mine!=generation:return
                frame_id=frame.id
                from explorer3.profiles import PROFILES
                profiles={k:k for k,v in PROFILES.items() if v['robot']==frame.robot}
                profile_select.set_options(profiles,value=next(iter(profiles)))
                shadow_button.enable()
                image.set_source('data:image/png;base64,'+base64.b64encode((DATA/frame.artifact_ref).read_bytes()).decode())
                details.clear()
                with details:
                    ui.label(frame.robot.upper()+' · '+frame.frame_id)
                    ui.label('Исходный кадр: '+frame.sample_id+' · записан ранее; текущее состояние неизвестно').classes('muted')
                    if streams:ui.label('Доступные потоки: '+', '.join(streams))
                    ui.label('Общая 3D-позиция не определена. Кадр не является подтверждением успеха.').classes('muted')
                status.set_text('Кадр проверен и сохранён в памяти с исходным временем и контрольной суммой.')
                detector_button.enable();segment_button.enable();model_button.enable()
            except Exception as exc:status.set_text(str(exc))
            finally:busy=False
        async def launch(method):
            nonlocal active,last_result,busy
            mine=generation;owns_busy=False
            try:
                if active or busy:raise ValueError('Дождитесь текущего разбора или отмените его')
                if not frame_id:raise ValueError('Сначала откройте кадр')
                if method=='vlm' and not ambiguous.value:raise ValueError('VLM используется только для неоднозначного кадра')
                busy=True;owns_busy=True
                ident=await asyncio.to_thread(jobs.cognitive_observation,[frame_id],question.value,method)
                if mine!=generation:
                    await asyncio.to_thread(jobs.cancel,ident);return
                active=ident
                last_result=None;status.set_text('Разбор поставлен в общую очередь. Роботы не подключаются.')
            except Exception as exc:status.set_text(str(exc))
            finally:
                if owns_busy:busy=False
        async def stop():
            nonlocal active,generation
            generation+=1
            if active:
                ident=active;active=None;await asyncio.to_thread(jobs.cancel,ident)
                status.set_text('Разбор отменён. Самостоятельного повторения нет.')
        ui.button('Открыть кадр',icon='image',on_click=open_frame)
        with ui.expansion('Получить один новый кадр с включённого робота').classes('w-full'):
            robot=ui.select({'rosmaster':'ROSMASTER','tank':'Tank'},value='rosmaster',label='Робот')
            ui.label('Используется уже работающий наблюдатель датчиков. Управление и моторный порт не открываются.').classes('muted')
            async def capture():
                nonlocal frame_id,busy
                mine=generation;owns_busy=False
                try:
                    if busy or active:raise ValueError('Сначала закончите или отмените текущий разбор')
                    busy=True;owns_busy=True;frame_id=None
                    for button in (shadow_button,detector_button,segment_button,model_button):button.disable()
                    from explorer3.profiles import PROFILES
                    frame=await asyncio.to_thread(observer.capture_readonly,robot.value)
                    if mine!=generation:return
                    frame_id=frame.id
                    image.set_source('data:image/png;base64,'+base64.b64encode((DATA/frame.artifact_ref).read_bytes()).decode())
                    options={k:k for k,v in PROFILES.items() if v['robot']==frame.robot}
                    profile_select.set_options(options,value=next(iter(options)))
                    details.clear()
                    with details:ui.label('Один полученный кадр · '+frame.robot+' · состояние быстро устаревает.').classes('muted')
                    for button in (shadow_button,detector_button,segment_button,model_button):button.enable()
                    status.set_text('Кадр получен. Команды движения не отправлялись.')
                except Exception as exc:status.set_text(str(exc))
                finally:
                    if owns_busy:busy=False
            ui.button('Получить кадр',on_click=capture)
        profile_select=ui.select({},label='Задача для проверки готовности').classes('w-full')
        async def prepare_real():
            try:
                from .real_preparation import prepare
                ident,result=await asyncio.to_thread(prepare,profile_select.value,[frame_id])
                details.clear()
                with details:
                    ui.label('Проверка подготовки · '+result['robot'].upper()).classes('sim-title')
                    for reason in result['problems']:ui.label(reason).classes('muted')
                    ui.link('Управление и запись',result['control_url'])
                    ui.button('Скачать проверку',on_click=lambda:ui.download(store.encoded(result).encode(),'readiness-'+ident[:8]+'.json'))
                status.set_text('Подготовка проверена. Управление не включалось.')
            except Exception as exc:status.set_text(str(exc))
        shadow_button=ui.button('Проверить подготовку задачи',on_click=prepare_real);shadow_button.disable()
        question=ui.input('Объект или короткий вопрос',value='кубик').classes('w-full')
        with ui.row():
            detector_button=ui.button('Найти объект · YOLOE',on_click=lambda:launch('yoloe'))
            segment_button=ui.button('Уточнить контур · SAM3',on_click=lambda:launch('sam3'))
        ambiguous=ui.checkbox('Кадр неоднозначен; нужен разбор локальной VLM',value=False)
        model_button=ui.button('Разобрать выбранный кадр · VLM',on_click=lambda:launch('vlm'))
        for button in (detector_button,segment_button,model_button):button.disable()
        ui.button('Отменить разбор',color='negative',on_click=stop)
        ui.label('Кнопки анализа запускают модель, но не обучение. Пока действует запрет вызовов моделей, используйте только «Открыть кадр».').classes('muted')
        def tracked_options():
            return {j['id']:j['id'][:8]+' · '+j['profile'] for j in catalog.rows('jobs')
                if j['status']=='completed' and json.loads(j['config']).get('kind')=='cognitive_observation'
                and json.loads(j['config']).get('method')=='yoloe'}
        with ui.expansion('Связать обнаружения соседних кадров').classes('w-full'):
            ui.label('ByteTrack: 2–32 ранее разобранных кадра одной камеры с интервалом до секунды. Идентификаторы остаются локальными.').classes('muted')
            detections=ui.select(tracked_options(),multiple=True,value=[],label='Готовые разборы YOLOE').classes('w-full')
            async def track():
                nonlocal active,busy
                mine=generation;owns_busy=False
                try:
                    if active or busy:raise ValueError('Дождитесь завершения разбора')
                    busy=True;owns_busy=True
                    ident=await asyncio.to_thread(jobs.cognitive_tracking,detections.value)
                    if mine!=generation:
                        await asyncio.to_thread(jobs.cancel,ident);return
                    active=ident
                    status.set_text('Трекинг сохранённых обнаружений поставлен в очередь.')
                except Exception as exc:status.set_text(str(exc))
                finally:
                    if owns_busy:busy=False
            ui.button('Связать обнаружения',on_click=track)
        ui.context.client.on_disconnect(stop)
        ui.on('observation_hidden',stop)
        ui.add_body_html("<script>document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('observation_hidden')});</script>")
        async def refresh():
            nonlocal active,last_result
            if active:
                job=await asyncio.to_thread(jobs.get,active)
                if job['status'] not in ('queued','running'):
                    ident=active;active=None;path=DATA/'jobs'/ident/'result.json'
                    if job['status']=='completed' and path.is_file():
                        last_result=json.loads(path.read_text());details.clear()
                        with details:
                            ui.label('Предложение модели · требует проверки; исход задачи неизвестен.').classes('sim-status')
                            answer=last_result.get('proposal',last_result.get('grounded',{}))
                            ui.label(answer.get('description') or json.dumps(answer,ensure_ascii=False)).style('overflow-wrap:anywhere')
                            if last_result.get('tracker'):
                                for frame in last_result['frames']:ui.label(frame['evidence_id'][:8]+': '+str(len(frame['tracks']))+' видимых треков')
                            ui.link('Файлы и источник','/jobs/'+ident)
                            ui.button('Скачать разбор',on_click=lambda:ui.download(store.encoded(last_result).encode(),'observation.json'))
                        status.set_text('Разбор завершён. Движение не разрешалось.')
                        detections.set_options(tracked_options())
                    else:status.set_text(job.get('error') or job['status'])
        ui.timer(1,refresh)

    @ui.page('/team/comparison')
    async def comparison():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f')
        ui.link('← Команда','/team');ui.html('<h1>Сравнение способов помощи</h1>')
        ui.label('Одинаковые навыки, расстановки и исходная память для четырёх режимов. Подготовка ничего не запускает.').classes('muted')
        from explorer3.sim.room import PRESETS
        scene=ui.select({k:v['label'] for k,v in PRESETS.items()},value='workshop',label='Окружение')
        start=ui.number('Первая расстановка',value=10000,min=0,precision=0)
        count=ui.number('Независимые расстановки',value=10,min=1,max=100,precision=0)
        split=ui.select({'validation':'Подбор метода','test':'Итоговая проверка'},value='validation',label='Назначение')
        status=ui.label('План ещё не подготовлен.').classes('sim-status');panel=ui.column().classes('w-full')
        plans=sorted((DATA/'cognitive/comparisons').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        saved=ui.select({p.stem:p.stem[:12] for p in plans},label='Сохранённый план').classes('w-full')
        async def report():
            try:
                result=await asyncio.to_thread(experiments.saved_report,saved.value)
                panel.clear()
                with panel:
                    ui.label('Все ячейки завершены' if result['complete'] else 'Серия не завершена или сопоставимость не подтверждена')
                    for row in result['rows']:
                        ui.label(f"{row['method']}: {row['reported']}/{row['planned']}; успех {row['success']}; неуспех {row['failure']}; технические/неизвестные {row['unknown_or_technical']}")
                    ui.button('Скачать отчёт',on_click=lambda:ui.download(store.encoded(result).encode(),'comparison-report.json'))
            except Exception as exc:status.set_text(str(exc))
        ui.button('Показать отчёт',on_click=report)
        async def prepare():
            try:
                seeds=list(range(int(start.value),int(start.value)+int(count.value)))
                ident,plan=await asyncio.to_thread(experiments.prepare,seeds,scene.value,split.value)
                saved.set_options({**saved.options,ident:ident[:12]},value=ident)
                panel.clear()
                with panel:
                    ui.label(f"{len(seeds)} расстановок × 4 режима · память зафиксирована · веса не меняются")
                    ui.label('M0 — правила; M1 — модель; M2 — память; M3 — память с проверкой совместимости.').classes('muted')
                    ui.label('Каждый запуск выполняется отдельно, автоматической длинной серии нет.').classes('muted')
                    ui.button('Скачать план',on_click=lambda:ui.download(store.encoded(plan).encode(),'comparison-'+ident[:8]+'.json'))
                    cell=ui.select({c['id']:str(c['seed'])+' · '+c['comparison'] for c in plan['cells']},value=plan['cells'][0]['id'],label='Один опыт')
                    approved=ui.checkbox('Разрешаю один симуляционный опыт с моделью, без обучения',value=False)
                    async def launch():
                        try:
                            if not approved.value:raise ValueError('Подтвердите выбранный опыт')
                            job=await asyncio.to_thread(jobs.cognitive_comparison,ident,cell.value)
                            status.set_text('Опыт: '+job);ui.navigate.to('/team?job='+job)
                        except Exception as exc:status.set_text(str(exc))
                    ui.button('Выполнить один опыт',on_click=launch).bind_enabled_from(approved,'value')
                status.set_text('План сохранён. Обучение и симуляция не запущены.')
            except Exception as exc:status.set_text(str(exc))
        ui.button('Подготовить сравнение · не запускать',icon='fact_check',on_click=prepare)
