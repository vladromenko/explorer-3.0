"""Common observation health and links to the existing independent controls."""
from .fleet_live import read_hub_status,readiness


def install(stop_all,css):
    from nicegui import app,ui
    @app.get('/api/fleet/live')
    def status():
        value=read_hub_status();return {**value,'readiness':readiness(value)}

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
        with ui.expansion('Подключение для разработчика').classes('workspace'):
            ui.label('ROS 2 Jazzy · отдельный домен 73 · доступ на Homebrain')
            ui.label('/explorer3/robots/tank/* · /explorer3/robots/rosmaster/* · /explorer3/fleet/status')
            ui.label('Штатные домены роботов и команды движения не перенастраиваются. Шина предназначена для наблюдений, а не для моторного управления.')
        def refresh():
            value=read_hub_status();ready=readiness(value)
            pair=value.get('pairing',{})
            pairing.set_text('Кадры двух роботов: '+pair.get('reason','Ожидание данных')+
                (f" · расхождение с учётом погрешности ≤ {pair['worst_case_skew_ms']} мс" if 'worst_case_skew_ms' in pair else ''))
            overall.set_text('Данные обеих машин доступны · время согласовано в пределах 50 мс' if ready['observations_ready']
                             else ' · '.join(ready['problems']))
            for robot in labels:
                state=value.get('robots',{}).get(robot,{})
                error=state.get('error') or (None if value.get('running') else 'Нет связи с общей шиной')
                labels[robot].set_text(error or f"Погрешность сопоставления времени: ≤ {state.get('uncertainty_ms','—')} мс (оценка)")
                names={'rgb':'Камера','depth':'Глубина','scan':'Лидар','scan_rear':'Второй лидар','imu':'IMU',
                       'odometry':'Одометрия','arm':'Суставы','mag':'Магнитометр','battery':'Питание'}
                tables[robot].rows=[{'name':name,'age':f'{state["ages_s"][key]:.2f} с'} for key,name in names.items() if key in state.get('ages_s',{})]
                tables[robot].update()
        ui.timer(1,refresh);refresh()
