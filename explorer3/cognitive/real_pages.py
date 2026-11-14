"""Explicit real-mission preparation and operator-controlled execution."""
import asyncio
import json
from nicegui import ui
from explorer3 import catalog,trials
from explorer3.paths import DATA
from explorer3.profiles import display_name
from explorer3.sim_pages import STYLE
from . import store,real_execution


def install(css,session=None,jobs=None):
    @ui.page('/team/real')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        ui.link('← Команда','/team');ui.html('<h1>Задача на роботах</h1>')
        ui.label('Последовательность уже подготовленных испытаний. Каждое движение требует вашего допуска; неизвестный исход останавливает продолжение.').classes('muted')
        executor=None
        if session is not None:
            from .physical_adapter import ExistingTrials
            executor=real_execution.Executor(ExistingTrials(session))
        generation=0;busy=False;plan_id=None;plan=None;shown=None;model_job=None;proposal=None
        options={}
        for path in sorted((DATA/'trials').glob('*.json')):
            value=json.loads(path.read_text())
            options[value['id']]=display_name(value['profile'])+' · '+('Повтор записи' if value['kind']=='replay' else 'Политика')+' · '+value['id'][:8]
        selected=ui.select(options,multiple=True,value=[],label='Испытания в порядке выполнения').classes('w-full')
        ui.label('Испытание сначала готовится в прежнем разделе «Испытания». Эта страница не создаёт моторную политику и не заменяет обучение.').classes('muted')
        ui.link('Открыть кадры и датчики для контекста','/team/observations')
        with catalog.connect() as db:
            rows=db.execute("SELECT body FROM cognitive_evidence WHERE json_extract(body,'$.domain')='real' ORDER BY expires DESC LIMIT 100").fetchall()
        items=[json.loads(r['body']) for r in rows]
        snapshots=ui.select({e['id']:e['robot'].upper()+' · '+('Архивный кадр' if e['payload'].get('archived') else 'Наблюдение')+' · '+e['id'][:8] for e in items},multiple=True,value=[],label='Исходные наблюдения каждого робота').classes('w-full')
        task=ui.input('Задача',value='Выполнить выбранные задачи и проверить каждый результат').classes('w-full')
        with ui.expansion('Использовать случай помощи другой платформы').classes('w-full'):
            memory=ui.select({'':'Без переноса',**{c['id']:c['recipient']+' → новый получатель · '+c['effect']+' · '+c['id'][:8]
                for c in store.cases() if c['domain']=='real'}},value='',label='Исходный случай памяти').classes('w-full')
            ui.label('Получателем будет последний шаг. Чужая память остаётся гипотезой: собственная версия навыка, камера и результат проверяются заново. Команды и веса между роботами не копируются.').classes('muted')
        status=ui.label('Подготовка не включает управление.').classes('sim-status');detail=ui.column().classes('w-full')
        confirmed=ui.checkbox('Я рядом; зарядники отключены, рабочие зоны свободны. Разрешаю только показанную миссию на 10 минут.',value=False)
        def invalidate():
            nonlocal plan_id,plan,proposal,generation
            plan_id=None;plan=None;proposal=None;generation+=1;confirmed.set_value(False)
            detail.clear();status.set_text('Выбор изменился. Подготовьте новый план перед запуском.')
        for field in (selected,snapshots,task,memory):field.on_value_change(invalidate)
        def display_plan():
            detail.clear()
            with detail:
                for node in plan['graph']['nodes']:
                    entry=plan['trials'][node['id']]
                    ui.label(str(plan['graph']['nodes'].index(node)+1)+'. '+display_name(entry['profile'])+' · '+('Повтор записи' if entry['kind']=='replay' else 'Политика'))
                if plan.get('memory_transfer'):
                    value=plan['memory_transfer']
                    ui.label('Перенос: '+value['donor']+' → '+value['recipient']+' · прежняя совместимость не переносится автоматически.').classes('muted')
                ui.button('Скачать точный план',on_click=lambda:ui.download(store.encoded(plan).encode(),'real-mission.json'))
        async def prepare():
            nonlocal plan_id,plan,proposal,busy
            mine=generation;owns_busy=False
            try:
                if busy or model_job or executor and executor.claim:raise ValueError('Сначала закончите или отмените текущую задачу')
                busy=True;owns_busy=True
                plan_id=None;plan=None;proposal=None;confirmed.set_value(False)
                ident,value=await asyncio.to_thread(real_execution.prepare,selected.value,snapshots.value,task.value,memory_case_id=memory.value or None)
                if mine!=generation:return
                plan_id=ident;plan=value
                display_plan()
                status.set_text('План сохранён. Реальные движения ещё не разрешены.')
            except Exception as exc:status.set_text(str(exc))
            finally:
                if owns_busy:busy=False
        async def stop(reason='STOP'):
            nonlocal generation,model_job,proposal
            generation+=1;confirmed.set_value(False)
            proposal=None
            if model_job:
                ident=model_job;model_job=None
                await asyncio.to_thread(jobs.cancel,ident)
            if executor:await asyncio.to_thread(executor.cancel,reason)
            status.set_text('Остановлено. Самостоятельного возобновления нет.')
        async def start():
            nonlocal busy
            mine=generation;owns_busy=False
            try:
                if busy or model_job:raise ValueError('Дождитесь текущего запроса или отмените его')
                if not executor:raise ValueError('В проверочном процессе физический исполнитель отсутствует')
                if not plan_id or not confirmed.value:raise ValueError('Подготовьте и подтвердите конкретный план')
                busy=True;owns_busy=True;confirmed.set_value(False)
                value=await asyncio.to_thread(real_execution.load,plan_id)
                for entry in value['trials'].values():await asyncio.to_thread(trials.authorize,entry['trial_id'],1/6)
                if mine!=generation:raise ValueError('Запуск отменён')
                ident=await asyncio.to_thread(executor.start,plan_id,operator_confirmed=True)
                status.set_text('Миссия '+ident[:8]+' · результат каждого шага отметьте ниже')
            except Exception as exc:status.set_text(str(exc))
            finally:
                if owns_busy:busy=False
        async def finish(outcome):
            try:
                if not executor:raise ValueError('Исполнитель недоступен')
                await asyncio.to_thread(executor.finish_step,outcome)
                status.set_text('Сохранение исходного эпизода; следующий робот пока не запускается.')
            except Exception as exc:status.set_text(str(exc))
        async def advance():
            try:
                if not executor:raise ValueError('Исполнитель недоступен')
                await asyncio.to_thread(executor.advance)
            except Exception as exc:status.set_text(str(exc))
        ui.button('Подготовить последовательность · не запускать',on_click=prepare)
        with ui.expansion('Локальный координатор · предложить порядок').classes('w-full'):
            ui.label('Один запрос Qwen по выбранным навыкам и источникам. Предложение требует просмотра; архив не подтверждает текущее состояние робота. Пока вызовы моделей запрещены, эту кнопку не нажимайте.').classes('muted')
            async def suggest():
                nonlocal model_job,proposal,busy
                mine=generation;owns_busy=False
                try:
                    if busy or model_job or executor and executor.claim:raise ValueError('Сначала завершите текущую задачу')
                    if not plan_id:raise ValueError('Сначала подготовьте конкретный план')
                    if jobs is None:raise ValueError('Очередь недоступна')
                    busy=True;owns_busy=True;proposal=None;confirmed.set_value(False)
                    ident=await asyncio.to_thread(jobs.cognitive_order,plan_id)
                    if mine!=generation:
                        await asyncio.to_thread(jobs.cancel,ident);return
                    model_job=ident;status.set_text('Координатор готовит предложение. Управление не включалось.')
                except Exception as exc:status.set_text(str(exc))
                finally:
                    if owns_busy:busy=False
            async def apply():
                nonlocal plan_id,plan,proposal
                mine=generation
                try:
                    if not proposal or busy or model_job or executor and executor.claim:raise ValueError('Нет завершённого предложения для просмотра')
                    ident,value=await asyncio.to_thread(real_execution.apply_order,plan_id,proposal)
                    if mine!=generation:return
                    plan_id=ident;plan=value;proposal=None;confirmed.set_value(False);display_plan()
                    status.set_text('Предложение перенесено в новый план. Проверьте шаги; движение не запускалось.')
                except Exception as exc:status.set_text(str(exc))
            ui.button('Запросить предложение модели',on_click=suggest)
            ui.button('Применить предложение к плану · не запускать',on_click=apply)
        start_button=ui.button('Разрешить эту миссию и начать',on_click=start).bind_enabled_from(confirmed,'value')
        if executor is None:start_button.disable()
        with ui.row():
            ui.button('Шаг выполнен · успех',on_click=lambda:finish('success'))
            ui.button('Шаг не выполнен',on_click=lambda:finish('failure'))
            ui.button('Следующий шаг · зона свободна',on_click=advance)
            ui.button('STOP · миссия',color='negative',on_click=lambda:stop())
        ui.label('Tank сохраняет блокировку USB/STOP. Подготовленный план её не снимает. Прежние панели и возврат управления доступны по обычным ссылкам Explorer.').classes('muted')
        ui.on('real_team_heartbeat',lambda:executor.heartbeat() if executor else None)
        ui.on('real_team_hidden',lambda:stop('Вкладка скрыта'))
        ui.context.client.on_disconnect(lambda:stop('Браузер отключился'))
        ui.add_body_html("<script>setInterval(()=>{if(!document.hidden)emitEvent('real_team_heartbeat')},400);document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('real_team_hidden')});window.addEventListener('pagehide',()=>emitEvent('real_team_hidden'));</script>")
        async def refresh():
            nonlocal shown,model_job,proposal
            if model_job:
                value=await asyncio.to_thread(jobs.get,model_job)
                if value['status'] not in ('queued','running'):
                    ident=model_job;model_job=None
                    path=DATA/'jobs'/ident/'result.json'
                    if value['status']=='completed' and path.is_file():
                        candidate=json.loads(path.read_text())
                        if candidate.get('source_plan_id')==plan_id:
                            proposal=candidate;status.set_text('Предложение: '+proposal['proposal']['reason'])
                        else:status.set_text('Выбор изменился; прежнее предложение не применяется.')
                    else:status.set_text(value.get('error') or value['status'])
            if executor:
                try:
                    await asyncio.to_thread(executor.tick)
                    if executor.state:
                        summary=(executor.state['phase'],executor.state['stage'],executor.state['index'],executor.state.get('reason'))
                        if summary!=shown:
                            labels={'running':'Выполняется','completed':'Завершено','cancelled':'Остановлено',
                                'starting':'Подключение','executing':'Испытание','saving':'Сохранение записи',
                                'ready_for_next':'Можно перейти к следующему шагу','done':'Все шаги приняты','stopped':'Продолжение отменено'}
                            shown=summary;status.set_text(f"{labels.get(summary[0],summary[0])} · {labels.get(summary[1],summary[1])} · завершено шагов {summary[2]}"+((' · '+summary[3]) if summary[3] else ''))
                except Exception as exc:
                    await stop(str(exc))
        ui.timer(.5,refresh)
