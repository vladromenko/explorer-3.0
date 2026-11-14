"""Review and bounded collection for R4 through the existing Jobs queue."""
import asyncio
import json
from pathlib import Path
from nicegui import ui
from . import catalog,noise_demonstrations as noise
from .paths import DATA
from .sim_pages import STYLE


def install(css,jobs):
    @ui.page('/sim/recovery-data')
    async def page():
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',negative='#d45050')
        ui.link('← Команда','/team');ui.html('<h1>Восстановительные показы</h1>')
        ui.label('DART-inspired · две симуляционные серии с одинаковым бюджетом. Масштаб возмущения берётся из измеренной ошибки ACT. Чистая команда эксперта и исполненная команда сохраняются раздельно.').classes('muted')
        sources={j['id']:j['profile']+' · '+j['id'][:8] for j in catalog.rows('jobs') if j['status']=='completed'
            and j['policy']=='act' and j['profile'].endswith('_sim') and not json.loads(j['config']).get('smoke',True)}
        if not sources:ui.label('Сначала нужен обычный симуляционный ACT и его независимая validation. Диагностические веса не подходят. Подготовка проекта не запускает это обучение.').classes('notice')
        source=ui.select(sources,label='ACT с измеренной ошибкой').classes('w-full')
        seed=ui.number('Первая новая расстановка',value=20000,min=0,precision=0)
        count=ui.number('Пар расстановок',value=3,min=1,max=20,precision=0)
        status=ui.label('Подготовка ничего не запускает.').classes('sim-status')
        plan_id=None;active=None;generation=0;busy=False
        confirmed=ui.checkbox('Разрешаю только показанный сбор в симуляции, без обучения',value=False)
        def invalidate():
            nonlocal plan_id,generation
            generation+=1;plan_id=None;confirmed.set_value(False)
            status.set_text('Выбор изменился. Подготовьте новый план.')
        for field in (source,seed,count):field.on_value_change(invalidate)
        async def prepare():
            nonlocal plan_id,busy
            mine=generation;owns_busy=False
            try:
                if active or busy:raise ValueError('Сначала завершите текущий сбор')
                busy=True;owns_busy=True
                plan_id=None;confirmed.set_value(False)
                ident,value=await asyncio.to_thread(noise.prepare,source.value,list(range(int(seed.value),int(seed.value)+int(count.value))))
                if mine!=generation:return
                plan_id=ident
                status.set_text('План '+plan_id[:8]+' · '+str(len(value['seeds']))+' чистых и столько же восстановительных показов. Движение и обучение не запускались.')
            except Exception as exc:status.set_text(str(exc))
            finally:
                if owns_busy:busy=False
        async def start():
            nonlocal active,busy
            mine=generation;owns_busy=False
            try:
                if busy or active or not plan_id or not confirmed.value:raise ValueError('Подготовьте и подтвердите точный план')
                busy=True;owns_busy=True;confirmed.set_value(False)
                ident=await asyncio.to_thread(jobs.collect_noise_demonstrations,plan_id)
                if mine!=generation:
                    await asyncio.to_thread(jobs.cancel,ident);return
                active=ident;status.set_text('Сбор в очереди. Обучение не запускается.')
            except Exception as exc:status.set_text(str(exc))
            finally:
                if owns_busy:busy=False
        async def stop():
            nonlocal active,generation
            generation+=1;confirmed.set_value(False)
            if active:
                ident=active;active=None;await asyncio.to_thread(jobs.cancel,ident)
            status.set_text('Отменено; самоповторения нет.')
        ui.button('Подготовить план · не запускать',on_click=prepare)
        ui.button('Собрать две серии в симуляции',on_click=start).bind_enabled_from(confirmed,'value')
        ui.button('Отменить сбор',color='negative',on_click=stop)
        ui.context.client.on_disconnect(stop);ui.on('noise_hidden',stop)
        ui.add_body_html("<script>document.addEventListener('visibilitychange',()=>{if(document.hidden)emitEvent('noise_hidden')});</script>")
        def choices():return {j['id']:j['id'][:8]+' · '+j['status'] for j in catalog.rows('jobs') if json.loads(j['config']).get('kind')=='noise_demonstrations'}
        saved=ui.select(choices(),label='Сохранённый сбор').classes('w-full');results=ui.column().classes('w-full')
        def show():
            try:
                if not saved.value or jobs.get(saved.value)['status']!='completed':raise ValueError('Выберите завершённый сбор')
                source_id=saved.value;report=json.loads((DATA/'jobs'/source_id/'result.json').read_text());results.clear()
                with results:
                    for index,item in enumerate(report['episodes']):
                        with ui.column().classes('sim-panel'):
                            ui.label(str(item['seed'])+' · '+item['arm']+' · '+item['outcome']+' · '+str(item['frames'])+' кадров')
                            paths=sorted(Path(item['spool']).glob('*.png'))
                            if paths:
                                with ui.row():
                                    for p in (paths[0],paths[len(paths)//2],paths[-1]):ui.image(p).style('width:180px')
                            if item['outcome']=='success':
                                reviewed=ui.checkbox('Проверил намерение эксперта, кадры и результат',value=False)
                                async def accept(i=index,check=reviewed):
                                    try:
                                        ident=await asyncio.to_thread(jobs.accept_noise_demonstration,source_id,i,check.value)
                                        status.set_text('Импорт принятого показа в очереди · '+ident[:8])
                                    except Exception as exc:status.set_text(str(exc))
                                ui.button('Принять отдельный показ',on_click=accept)
                    ui.link('Подготовить обучение по принятым показам','/training/prepare')
                    ui.label('Сравнивайте одинаковые бюджеты ACT и отдельные test-seeds. Неуспешные показы остаются в отчёте; снижение ошибки само по себе не доказывает пользу восстановления.').classes('muted')
            except Exception as exc:status.set_text(str(exc))
        ui.button('Открыть результат',on_click=show)
        async def refresh():
            nonlocal active
            if active:
                value=await asyncio.to_thread(jobs.get,active)
                if value['status'] not in ('queued','running'):
                    ident=active;active=None;saved.set_options(choices(),value=ident)
                    status.set_text('Сбор завершён; откройте результат.' if value['status']=='completed' else value.get('error') or value['status'])
        ui.timer(1,refresh)
