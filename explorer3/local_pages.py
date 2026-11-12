"""Local job, dataset and research views. These pages never acquire hardware."""
import asyncio
import hashlib
import json
from pathlib import Path
from nicegui import app, ui
from . import catalog, research
from .paths import DATA
from .datasets import eligible, grouped_split


def install(jobs):
    @app.get('/api/research/export')
    def export_research():
        return {'studies': research.studies(), 'records': research.records(), 'physical_skill_verified': False}

    @ui.page('/jobs/{ident}')
    def job_page(ident):
        jobs.get(ident)
        ui.dark_mode().enable()
        ui.link('← Explorer 3.0', '/')
        ui.label('Задание ' + ident).style('font-size:24px')
        state = ui.label()
        report = ui.code('', language='json').style('width:100%;max-height:440px;overflow:auto')
        log = ui.code('', language='text').style('width:100%;max-height:500px;overflow:auto')
        with ui.row():
            ui.button('Отменить задание', on_click=lambda: jobs.cancel(ident))
            def resume():
                try:
                    jobs.resume(ident)
                    ui.notify('Продолжение поставлено в очередь')
                except Exception as exc:
                    ui.notify(str(exc),type='negative')
            ui.button('Продолжить с checkpoint', on_click=resume)
        def refresh():
            job = jobs.get(ident)
            cfg = json.loads(job['config'])
            state.set_text(job['status'] + ' · ' + cfg.get('kind','train') + (' · '+job['error'] if job['error'] else ''))
            log.set_content(jobs.log(ident))
            directory = DATA/'jobs'/ident
            results = {name: json.loads((directory/name).read_text()) for name in ('result.json','evaluation.json') if (directory/name).exists()}
            report.set_content(json.dumps(results or cfg, ensure_ascii=False, indent=2))
        refresh()
        ui.timer(3,refresh)

    @ui.page('/research')
    def research_page():
        ui.dark_mode().enable()
        ui.link('← Explorer 3.0','/')
        ui.label('Оценка полезности вмешательства').style('font-size:26px')
        ui.label('G = P(успех при назначенной коррекции) − P(успех без коррекции)')
        ui.label('Показы и smoke checkpoints не доказывают эффект помощи. Для обучения нужны отдельные рандомизированные эпизоды.')
        jobs_with_weights = {j['id']: j['profile']+' · '+j['id'][:8] for j in catalog.rows('jobs') if j['checkpoint'] and json.loads(j['config']).get('kind','train')=='train'}
        checkpoint = ui.select(jobs_with_weights,label='Замороженная политика для проекта пилота').style('min-width:400px')
        ui.label('Проект пилота: hold во время ожидания, одно назначение на эпизод, p=0.5, до 10 секунд коррекции. Задержки требуют физической проверки.')
        async def create():
            try:
                if not checkpoint.value:
                    raise ValueError('Выберите checkpoint')
                row = jobs.get(checkpoint.value)
                weights = Path(row['checkpoint'])/'model.safetensors'
                checksum = await asyncio.to_thread(lambda:hashlib.sha256(weights.read_bytes()).hexdigest())
                study = research.create_study(row['profile'],checksum)
                ui.notify('Сохранён проект пилота '+study[:8]+'; движения не запускаются')
                refresh()
            except Exception as exc:ui.notify(str(exc),type='negative')
        ui.button('Сохранить проект пилота',on_click=create)
        table=ui.table(columns=[{'name':k,'field':k,'label':v} for k,v in [('id','Пилот'),('profile','Профиль'),('status','Состояние'),('correction_limit_s','Лимит, с')]],rows=[],row_key='id')
        counts=ui.label()
        def analyze():
            try:
                ident=jobs.research()
                ui.navigate.to('/jobs/'+ident)
            except Exception as exc:ui.notify(str(exc),type='negative')
        ui.button('Обучить оценщики и сравнить перенос 0/10/20',on_click=analyze)
        ui.link('Экспорт записанного журнала (JSON)','/api/research/export',new_tab=True)
        ui.label('Сравниваются признаки с физическим ответом и без него; оба режима ROSMASTER исключены из исходного обучения. Нормировка — только train. Результаты группируются по эпизодам и сессиям.')
        def refresh():
            table.rows=research.studies();table.update()
            rows=research.records()
            counts.set_text(f"Назначений: {len(rows)} · завершённых: {sum(r['outcome'] in {'success','failure'} for r in rows)} · аварийных: {sum(r.get('emergency',False) for r in rows)}")
        refresh()


def collection_status(name):
    rows=eligible(name)
    try:
        splits=grouped_split(rows)
        grouping='Независимые группы: train / validation / test = '+ ' / '.join(str(len(v)) for v in (splits['train'],splits['validation'],splits['test']))
    except ValueError as exc:grouping=str(exc)
    return f'Качественных показов: {len(rows)} / 50. '+grouping


def profile_statuses():
    from .profiles import PROFILES
    jobs=catalog.rows('jobs')
    rows=[]
    for name,cfg in PROFILES.items():
        trained=[j for j in jobs if j['profile']==name and j['status']=='completed'
                 and json.loads(j['config']).get('kind','train')=='train']
        full=any(not json.loads(j['config']).get('smoke',True) for j in trained)
        rows.append({'profile':name,'record':'проверена' if cfg['recording_status']=='verified' else 'не проверено',
                     'train':'основное обучение завершено' if full else 'smoke проверен' if trained or cfg['training_status']=='smoke_verified' else 'не проверено',
                     'skill':'не проверено'})
    return rows
