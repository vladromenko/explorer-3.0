"""Preparation is separate from an explicit training start; no automatic runs."""
import asyncio
import json
from nicegui import ui
from . import catalog,datasets,training_drafts
from .paths import DATA,ROOT
from .profiles import PROFILES
from .policy_catalog import POLICIES,validate_recipe
from .sim_pages import STYLE


def install(css,jobs):
    @ui.page('/sim/training')
    async def page():
        await builder_page('sim')

    @ui.page('/training/prepare')
    async def real_page():
        await builder_page('real')

    @ui.page('/datasets/guide')
    async def guide_page():
        ui.add_css(css+STYLE);ui.dark_mode().enable()
        ui.add_css('.dataset-guide{min-width:0;overflow-wrap:anywhere}.dataset-guide table{display:block;max-width:100%;overflow-x:auto}.dataset-guide code{white-space:normal}')
        with ui.row():
            ui.link('Explorer','/');ui.link('Подготовить обучение','/training/prepare')
        ui.markdown((ROOT/'docs/DATASET_GUIDE_RU.md').read_text()).classes('w-full dataset-guide')

    async def builder_page(domain):
        ui.add_css(css+STYLE);ui.dark_mode().enable();ui.colors(primary='#2b9c8f',secondary='#748aa4',negative='#d45050')
        draft_id=None;preparing=False;loading=False;starting=False;selection_generation=0
        def refresh():
            available={}
            for ident,label in POLICIES.items():
                try:
                    validate_recipe(ident,profile.value);available[ident]=label
                except ValueError:pass
            if policy.options!=available:
                policy.set_options(available,value=policy.value if policy.value in available else 'act')
            shows={e['id']:e['id'][:8]+' · '+str(e['frames'])+' кадров · '+e['category'] for e in datasets.eligible(profile.value)}
            episodes.set_options(shows,value=[])
            sources={'new':'Новая модель · без прежнего checkpoint'}
            for j in catalog.rows('jobs'):
                if j['profile']==profile.value and j['policy']==policy.value and j['status']=='completed' and j['checkpoint'] and json.loads(j['config']).get('kind','train')=='train':
                    sources[j['id']]='Дообучить '+j['id'][:8]
            source.set_options(sources,value='new')
        async def prepare():
            nonlocal draft_id,preparing
            if preparing or loading or starting:return
            draft_id=None;confirmed.value=False;start_button.disable()
            preparing=True;prepare_button.disable();status.set_text('Проверка данных и сохранение плана · обучение не запущено')
            try:
                parent=None if source.value=='new' else jobs.get(source.value)
                draft_id,draft=await asyncio.to_thread(training_drafts.prepare,profile.value,policy.value,episodes.value,
                    steps=int(steps.value),batch=int(batch.value),seed=int(seed.value),device=device.value,source=parent)
                show(draft);history.set_options(saved(),value=draft_id);status.set_text('План сохранён · обучение не запущено')
            except Exception as exc:
                draft_id=None;confirmed.value=False;start_button.disable();status.set_text(str(exc))
            finally:preparing=False;prepare_button.enable()
        async def start():
            nonlocal starting
            if preparing or loading or starting:return
            try:
                if not confirmed.value:raise ValueError('Сначала подтвердите запуск выбранного плана')
                if draft_id is None:raise ValueError('Сначала подготовьте или откройте план')
                selected_draft=draft_id;starting=True;start_button.disable();confirmed.value=False
                ident=await asyncio.to_thread(jobs.start_prepared_training,selected_draft)
                confirmed.value=False;status.set_text('Обучение поставлено в очередь · '+ident[:8])
                with job_links:ui.link('Открыть задание · ход работы, отмена и продолжение','/jobs/'+ident)
            except Exception as exc:status.set_text(str(exc))
            finally:
                starting=False;start_button.set_enabled(draft_id is not None and not preparing and not loading)
        with ui.element('div').classes('hero'):
            with ui.column():
                ui.label('HOMEBRAIN / ПОДГОТОВКА ОБУЧЕНИЯ').classes('kicker');ui.html('<h1>Explorer 3.0</h1>')
                ui.label('Сначала проверить план, затем запустить самостоятельно').classes('muted')
            ui.label('Реальные записи' if domain=='real' else 'Симуляционные данные').classes('sim-badge')
        with ui.row():
            ui.link('← Explorer','/');ui.link('Как собрать датасет','/datasets/guide')
        names=({key:value.get('label',key) for key,value in PROFILES.items()} if domain=='real' else
            {'so101_pick_place_sim':'SO-101 · кубик в коробку','tank_dock_sim':'Tank · подъезд к цели',
             'rosmaster_pick_place_sim':'ROSMASTER · кубик в коробку','rosmaster_dock_sim':'ROSMASTER · подъезд к цели'})
        with ui.element('div').classes('sim-layout') as builder:
            with ui.column().classes('sim-panel'):
                ui.label('Данные и исходная модель').classes('sim-title')
                profile=ui.select(names,value=next(iter(names)),label='Задача',on_change=lambda:refresh()).classes('w-full')
                policy=ui.select(POLICIES,value='act',label='Модель',on_change=lambda:refresh()).classes('w-full')
                source=ui.select({},label='Начать заново или дообучить').classes('w-full')
                episodes=ui.select({},multiple=True,label='Показы и исправления для нового плана').props('use-chips').classes('w-full')
                ui.button('Обновить список данных',icon='refresh',on_click=refresh).props('outline')
                ui.label(('Нужны минимум 50 успешных пригодных показов одного профиля и минимум три независимые группы.' if domain=='real' else
                    'Выберите успешные записи одной версии физики, минимум три независимые группы.')+
                    ' Общие сессии ИЛИ расстановки не разделяются между обучением и проверкой. Количество не гарантирует навык.').classes('muted')
                ui.label('При дообучении прежние данные сохраняются. Сцены исходного обучения не попадут в проверку; прежние контрольные группы останутся контрольными.').classes('muted')
                refresh()
            with ui.column().classes('sim-panel'):
                ui.label('Конечный бюджет').classes('sim-title')
                steps=ui.number('Обновлений весов',value=2000,min=50,max=1000000,step=50,precision=0).classes('w-full')
                batch=ui.number('Примеров за одно обновление',value=1,min=1,max=32,precision=0).classes('w-full')
                seed=ui.number('Seed запуска',value=1000,min=0,precision=0).classes('w-full')
                device=ui.select({'cuda':'GPU Homebrain','cpu':'CPU Homebrain'},value='cuda',label='Вычисления').classes('w-full')
                ui.label('Бюджет задаёте вы. Размер пакета не уменьшается незаметно при нехватке памяти. Длительность заранее не обещается: она зависит от модели и данных.').classes('muted')
                prepare_button=ui.button('Подготовить план · не запускать',icon='description',on_click=prepare)
        def saved():
            directory=DATA/'training-drafts'
            return {p.stem:p.stem[:8]+' · '+json.loads(p.read_text())['policy'].upper() for p in sorted(directory.glob('*.json'))
                if json.loads(p.read_text()).get('profile','') in names
                and json.loads(p.read_text()).get('mode') in ('new_training','fine_tune_new_optimizer')} if directory.exists() else {}
        async def open_saved():
            nonlocal draft_id,loading,selection_generation
            if preparing or starting:return
            selected_draft=history.value
            if selected_draft is not None and selected_draft==draft_id:return
            selection_generation+=1;generation=selection_generation
            draft_id=None;confirmed.value=False;start_button.disable()
            summary.clear();summary.set_visibility(False);loading=bool(selected_draft)
            if selected_draft:
                status.set_text('Проверка выбранного плана · запуск недоступен')
                try:
                    draft=await asyncio.to_thread(training_drafts.load,selected_draft)
                    if generation!=selection_generation:return
                    draft_id=selected_draft
                    show(draft);status.set_text('Сохранённый план проверен · обучение не запущено')
                except Exception as exc:
                    if generation==selection_generation:draft_id=None;status.set_text(str(exc))
                finally:
                    if generation==selection_generation:
                        loading=False;start_button.set_enabled(draft_id is not None)
        history=ui.select(saved(),label='Открыть подготовленный план',on_change=open_saved).classes('w-full')
        def another():
            nonlocal draft_id,loading,selection_generation
            if preparing or starting:return
            selection_generation+=1;loading=False;start_button.disable()
            draft_id=None;summary.clear();summary.set_visibility(False);builder.set_visibility(True)
            history.value=None
            confirmed.value=False;status.set_text('Выберите параметры нового плана · обучение не запущено')
        ui.button('Подготовить другой план',icon='edit',on_click=another).props('outline')
        summary=ui.column().classes('sim-panel');summary.set_visibility(False)
        with ui.column().classes('sim-panel'):
            confirmed=ui.checkbox('Запустить обучение сейчас по показанному неизменяемому плану')
            start_button=ui.button('Запустить подготовленное обучение',icon='play_arrow',on_click=start)
            start_button.disable()
            status=ui.label('Подготовка не запускает обучение.').classes('sim-status')
            job_links=ui.column().classes('w-full')
            ui.label('Продолжение прерванного задания выполняется отдельно через его карточку: оно сохраняет optimizer и прежний план. Дообучение здесь создаёт новое задание из выбранных весов.').classes('muted')
        def show(draft):
            summary.clear();summary.set_visibility(True);builder.set_visibility(False);confirmed.value=False
            job_links.clear()
            with job_links:
                for task in catalog.rows('jobs'):
                    if json.loads(task['config']).get('draft_id')==draft_id:ui.link(task['status']+' · '+task['id'][:8],'/jobs/'+task['id'])
            with summary:
                ui.label('Готовый план · '+draft_id[:8]).classes('sim-title')
                ui.label(draft['policy'].upper()+' · '+names.get(draft['profile'],draft['profile'])+' · '+str(draft['steps'])+' обновлений · пакет '+str(draft['batch_size'])+' · '+draft['device'])
                ui.table(columns=[{'name':'part','label':'Часть данных','field':'part'},
                    {'name':'episodes','label':'Записей','field':'episodes'},{'name':'frames','label':'Кадров','field':'frames'}],
                    rows=[{'part':{'train':'Обучение','validation':'Проверка при обучении','test':'Итоговая проверка'}[role],
                        'episodes':count,'frames':draft['frames'][role]} for role,count in draft['split_counts'].items()],row_key='part').classes('w-full')
                mix=draft['mixture'];ui.label(f"Обучающие кадры: {mix['old_train_frames']} прежних + {mix['new_train_frames']} новых. Нормировка только по обучающей части.")
                ui.button('Скачать план',icon='download',on_click=lambda:ui.download.file(DATA/'training-drafts'/(draft_id+'.json'))).props('outline')
            start_button.enable()
