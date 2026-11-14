"""Common observation health and links to the existing independent controls."""
from .fleet_live import read_hub_status,readiness


def install(stop_all,css):
    from nicegui import app,ui
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from . import fleet_capture
    @app.get('/api/fleet/live')
    def status():
        value=read_hub_status();return {**value,'readiness':readiness(value)}

    @app.get('/api/fleet/captures')
    def captures():return fleet_capture.state()

    @app.get('/api/fleet/captures/{identifier}/{filename}')
    def capture_file(identifier:str,filename:str):
        try:
            root=fleet_capture.entry(identifier)
            import re
            if filename in ('manifest.json','clock-status.json','replay.json'):path=root/filename
            elif re.fullmatch(r'preview-\d{3}-(tank|rosmaster)-(rgb|depth|scan|scan_rear)\.jpg',filename):path=root/filename
            elif filename=='metadata.yaml' or (filename.startswith('bag_') and filename.endswith('.mcap') and '/' not in filename):path=root/'bag'/filename
            else:raise ValueError('Нет такого файла записи')
            if not path.is_file() or path.is_symlink():raise ValueError('Файл ещё не сохранён')
            return FileResponse(path,filename=filename)
        except ValueError as exc:raise HTTPException(404,str(exc)) from exc

    @ui.page('/fleet/captures/{identifier}')
    def replay_page(identifier:str):
        import json
        ui.add_css(css);ui.dark_mode().enable();ui.link('← Общий обзор','/fleet/live')
        ui.label('Сохранённая сессия').style('font-size:30px;font-weight:700')
        ui.label('Обе машины на одной временной шкале. Изображения взяты из сохранённого файла; подключения к роботам нет.').classes('muted')
        try:
            path=fleet_capture.entry(identifier)
            data=json.loads((path/'replay.json').read_text());frames=data['frames']
            if not frames:raise ValueError('Нет кадров для просмотра')
        except (ValueError,OSError,KeyError):ui.label('Для этой записи просмотр ещё не подготовлен');return
        ui.label('Камеры снимают в разное время. Показаны последние данные к выбранному моменту, возрастом не более 1,5 с.').classes('muted')
        first=next((i for i,f in enumerate(frames) if 'tank/rgb' in f['views'] and 'rosmaster/rgb' in f['views']),0)
        position=ui.label();slider=ui.slider(min=0,max=len(frames)-1,value=first).classes('w-full')
        playing=[False]
        def toggle():
            playing[0]=not playing[0]
            if playing[0] and slider.value==len(frames)-1:slider.set_value(0)
            play.set_text('Пауза просмотра' if playing[0] else 'Воспроизвести запись')
        with ui.row():
            ui.button('Назад',icon='skip_previous',on_click=lambda:slider.set_value(max(0,int(slider.value)-1)))
            play=ui.button('Воспроизвести запись',icon='play_arrow',on_click=toggle)
            ui.button('Вперёд',icon='skip_next',on_click=lambda:slider.set_value(min(len(frames)-1,int(slider.value)+1)))
        def tick():
            if not playing[0]:return
            if slider.value<len(frames)-1:slider.set_value(int(slider.value)+1)
            else:playing[0]=False;play.set_text('Воспроизвести запись')
        ui.timer(1,tick)
        images={};ages={}
        with ui.element('div').classes('fleet').style('grid-template-columns:repeat(auto-fit,minmax(280px,1fr));align-items:start'):
            for robot,title in [('tank','Tank'),('rosmaster','ROSMASTER')]:
                with ui.card().classes('robot'):
                    ui.label(title).style('font-size:24px;font-weight:650')
                    for key,label in [('rgb','Камера'),('depth','Глубина'),('scan','Лидар'),('scan_rear','Второй лидар')]:
                        if key=='scan_rear' and robot=='tank':continue
                        ui.label(label)
                        images[robot+'/'+key]=ui.image().classes('w-full').props('fit=contain no-spinner').style('height:280px;background:#10161f;border-radius:8px')
                        ages[robot+'/'+key]=ui.label().classes('muted')
        def show():
            frame=frames[int(slider.value)]
            position.set_text(f"{frame['after_s']:.1f} с / {frames[-1]['after_s']:.1f} с")
            for key,img in images.items():
                view=frame['views'].get(key);img.set_visibility(bool(view))
                if view:img.set_source(f"/api/fleet/captures/{identifier}/{view['file']}")
                ages[key].set_text(f"Возраст данных: {view['age_s']:.2f} с" if view else 'В этот момент свежих данных нет')
        slider.on_value_change(show);show()

    @ui.page('/fleet/live')
    def page():
        ui.add_css(css);ui.dark_mode().enable()
        ui.link('← Explorer 3.0','/')
        ui.label('ОБЩИЙ ОБЗОР').classes('kicker')
        ui.label('Роботы в одной системе').style('font-size:32px;font-weight:700')
        ui.label('Homebrain собирает датчики обеих машин. Управление каждой машиной остаётся в её панели.').classes('muted')
        ui.button('STOP · всё управление Explorer',on_click=stop_all,color='negative',icon='stop_circle')
        overall=ui.label('Проверка связи…').classes('notice')
        pairing=ui.label('Сопоставление кадров…').classes('muted')
        labels={};tables={}
        with ui.element('div').classes('fleet').style('grid-template-columns:repeat(auto-fit,minmax(280px,1fr))'):
            for robot,title in [('tank','Tank'),('rosmaster','ROSMASTER')]:
                with ui.card().classes('robot'):
                    ui.label(title).style('font-size:24px;font-weight:650')
                    labels[robot]=ui.label('Ожидание данных').classes('muted')
                    ui.link('Камеры и окружение',f'/{robot}/sensors')
                    ui.link('Открыть управление','/rosmaster/control' if robot=='rosmaster' else '/')
                    tables[robot]=ui.table(columns=[{'name':'name','label':'Датчик','field':'name'},
                        {'name':'age','label':'Возраст данных','field':'age'}],rows=[],row_key='name').props('dense flat')
        ui.label('Общая карта: требуется физическая калибровка').style('font-size:20px;font-weight:600')
        ui.label('Пока каждый робот видит окружение в собственных координатах. Их положение относительно друг друга не задано. '
                 'Согласование времени по сети не означает одновременную съёмку камер.').classes('muted')
        ui.label('Общие автономные движения ещё не прошли приёмку. Эта страница не запускает моторы и не включает обучение.').classes('muted')
        with ui.card().classes('workspace'):
            ui.label('Сохранить общую сессию').style('font-size:22px;font-weight:650')
            ui.label('Камеры, глубина, лидары и телеметрия в одном файле. Для разбора окружения; это не обучающий показ.').classes('muted')
            with ui.row().classes('items-center'):
                duration=ui.select({15:'15 секунд',30:'30 секунд',60:'1 минута'},value=30).props('outlined dense')
                def capture():
                    try:fleet_capture.start(duration.value);ui.notify('Общая запись началась')
                    except (ValueError,RuntimeError) as exc:ui.notify(str(exc),type='negative')
                capture_button=ui.button('Записать датчики',icon='fiber_manual_record',on_click=capture)
            capture_state=ui.label().classes('muted')
            @ui.refreshable
            def capture_list():
                import time
                for record in fleet_capture.records()[:10]:
                    identifier=record['id'];phase=record['phase']
                    label={'recording':'записывается','complete':'сохранена','failed':'ошибка','interrupted':'сохранение не подтверждено'}.get(phase,phase)
                    with ui.expansion(time.strftime('%d.%m %H:%M:%S',time.localtime(record['created_at']))+' · '+label).classes('w-full'):
                        if record.get('error'):ui.label('Запись не завершена. Подробности сохранены в её отчёте.').classes('muted')
                        streams=record.get('streams',{})
                        if streams:
                            ui.label(f"{sum(s['count'] for s in streams.values())} сообщений · повторное чтение проверено")
                            ui.table(columns=[{'name':'stream','label':'Поток','field':'stream'},
                                {'name':'count','label':'Измерений','field':'count'}],
                                rows=[{'stream':k.replace('/explorer3/robots/',''),'count':v['count']} for k,v in streams.items()],row_key='stream').props('dense flat')
                        if phase=='complete':
                            if record.get('replay_frames'):ui.link('Смотреть вместе',f'/fleet/captures/{identifier}')
                            ui.link('Отчёт о записи',f'/api/fleet/captures/{identifier}/manifest.json',new_tab=True)
                            ui.link('Согласование времени',f'/api/fleet/captures/{identifier}/clock-status.json',new_tab=True)
                            for path in sorted((fleet_capture.entry(identifier)/'bag').glob('*.mcap')):
                                ui.link('Скачать '+path.name,f'/api/fleet/captures/{identifier}/{path.name}',new_tab=True)
            capture_list()
            capture_version=[None]
            def refresh_capture():
                state=fleet_capture.state();active=state['active']
                capture_button.set_enabled(not active)
                capture_state.set_text('Записываем данные. Завершение автоматическое; управление роботами не меняется.' if active else 'Готово к записи доступных датчиков')
                version=tuple((r['id'],r['phase']) for r in state['records'])
                if version!=capture_version[0]:capture_version[0]=version;capture_list.refresh()
            ui.timer(1,refresh_capture)
        with ui.expansion('Подключение для разработчика').classes('workspace'):
            ui.label('ROS 2 Jazzy · отдельный домен 73 · доступ на Homebrain')
            ui.label('/explorer3/robots/tank/* · /explorer3/robots/rosmaster/* · /explorer3/fleet/status')
            ui.label('Штатные домены роботов и команды движения не перенастраиваются. Шина предназначена для наблюдений, а не для моторного управления.')
        def refresh():
            value=read_hub_status();ready=readiness(value)
            pair=value.get('pairing',{})
            pairing.set_text('Кадры двух роботов: '+pair.get('reason','Ожидание данных')+
                (f" · расхождение с учётом погрешности ≤ {pair['worst_case_skew_ms']} мс" if 'worst_case_skew_ms' in pair else ''))
            if not value.get('running'):overall.set_text('Общая шина временно недоступна. Ниже — последние полученные данные.')
            elif ready['observations_ready']:overall.set_text('Обе машины подключены · данные доступны вместе')
            else:
                parts=[]
                for robot,title in [('tank','Tank'),('rosmaster','ROSMASTER')]:
                    one=readiness(value,required=(robot,))
                    parts.append(title+(' подключён' if one['observations_ready'] else ' — ожидаем данные'))
                overall.set_text(' · '.join(parts))
            for robot in labels:
                state=value.get('robots',{}).get(robot,{})
                error=state.get('error') or (None if value.get('running') else 'Нет связи с общей шиной')
                uncertainty=state.get('uncertainty_ms')
                shown=f'{uncertainty:.1f}' if isinstance(uncertainty,(float,int)) else '—'
                labels[robot].set_text(error or f"Погрешность сопоставления времени: ≤ {shown} мс (оценка)")
                names={'rgb':'Камера','depth':'Глубина','scan':'Лидар','scan_rear':'Второй лидар','imu':'IMU',
                       'odometry':'Одометрия','arm':'Суставы','mag':'Магнитометр','battery':'Питание'}
                tables[robot].rows=[{'name':name,'age':f'{state["ages_s"][key]:.2f} с'} for key,name in names.items() if key in state.get('ages_s',{})]
                tables[robot].update()
        ui.timer(1,refresh);refresh()
