"""Local sensor inspection, with no ownership or movement controls."""
import base64
import math
import time

import cv2
import numpy as np

from .sensors import SensorFeed,SensorArchive,stream_age,archive_entries,open_archive,archive_packet


def jpeg(image):
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,78])
    if not ok:raise ValueError('Cannot render sensor preview')
    return 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()


def previews(packet):
    arrays=packet['arrays'];streams=packet['meta']['streams'];result={}
    if 'rgb' in arrays:result['rgb']=jpeg(cv2.cvtColor(arrays['rgb'],cv2.COLOR_RGB2BGR))
    if 'ir' in arrays:
        ir=arrays['ir'];ir=np.clip(ir,0,255).astype(np.uint8)
        result['ir']=jpeg(ir)
    if 'depth' in arrays:
        raw=arrays['depth'];metric=raw.astype(float)*(.001 if streams['depth']['unit']=='mm' else 1.)
        valid=np.isfinite(metric)&(metric>=.15)&(metric<=4.)
        gray=np.clip(np.nan_to_num(metric,nan=0,posinf=0)*255/4,0,255).astype(np.uint8)
        colored=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);colored[~valid]=0
        result['depth']=jpeg(colored)
    if 'scan' in arrays:
        canvas=np.full((500,500,3),(31,22,16),np.uint8)
        for radius in (50,100,150,200):
            cv2.circle(canvas,(250,250),radius,(65,61,54),1)
            cv2.putText(canvas,f'{radius/50:g}m',(254,249-radius),cv2.FONT_HERSHEY_SIMPLEX,.35,(160,160,160),1)
        cv2.arrowedLine(canvas,(250,270),(250,220),(210,210,210),2)
        info=streams['scan'];ranges=arrays['scan']
        angles=info['angle_min']+np.arange(len(ranges))*info['angle_increment']
        valid=np.isfinite(ranges)&(ranges>=info['range_min'])&(ranges<=min(4.8,info['range_max']))
        for radius,angle in zip(ranges[valid],angles[valid]):
            point=(int(250-radius*math.sin(angle)*50),int(250-radius*math.cos(angle)*50))
            cv2.circle(canvas,point,2,(190,210,78),-1)
        result['scan']=jpeg(canvas)
    return result


def install():
    from nicegui import app,ui,run
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from .paths import DATA
    feed=SensorFeed(isolated=None);app.on_shutdown(feed.close)

    @app.get('/api/tank/sensors/audio/{ident}')
    def audio_file(ident:str):
        if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):raise HTTPException(404)
        path=DATA/'sensor-audio'/('tank-'+ident+'.wav')
        if not path.is_file():raise HTTPException(404)
        return FileResponse(path,media_type='audio/wav',headers={'Cache-Control':'no-store'})

    @ui.page('/tank/sensors')
    async def page(recording: str | None=None):
        ui.dark_mode().enable()
        ui.add_css('''body{background:#10161f;color:#e6edf4;font-family:system-ui,sans-serif}
          .nicegui-content{max-width:1440px;width:100%;margin:auto;padding:26px}
          .sensor-card{background:#192331;border:1px solid #344253;border-radius:14px;box-shadow:none;min-width:0;max-width:100%}
          .w-full{width:100%}.text-3xl{font-size:32px;font-weight:700}.text-lg{font-size:18px;font-weight:600}
          .text-xs{font-size:12px}.text-sm{font-size:13px}.text-teal-300,.text-teal-200{color:#8de3d6}
          .tracking-widest{letter-spacing:.12em}.nicegui-scene{width:100%!important;height:420px!important;overflow:hidden}
          .q-table__container{width:100%;max-width:100%;overflow:auto}.q-table td{white-space:normal}
          .sensor-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;width:100%}
          @media(max-width:800px){.sensor-grid{grid-template-columns:1fr}}
          .q-btn{border-radius:9px}.muted{color:#9aacbf}''')
        ui.link('← Explorer 3.0','/').classes('text-teal-300')
        replay_root=None;replay_rows=[];replay_packet=None;replay_index=-1
        if recording:
            try:replay_root,replay_rows=open_archive(recording)
            except (ValueError,OSError):
                ui.label('Запись датчиков не найдена');return
            if not replay_rows:ui.label('В записи нет пакетов');return
        ui.label('TANK · ОКРУЖЕНИЕ').classes('text-xs tracking-widest text-teal-300')
        ui.label('Что видит робот').classes('text-3xl font-bold')
        ui.label('Камеры, глубина и лазерный обзор. Подключение только для чтения: гусеницы не запускаются.').classes('muted')
        state=ui.label('Нажмите «Подключить датчики»').classes('text-teal-200')
        archive=None;last_packet=None;started=0.;busy=False
        def connect():feed.start();state.set_text('Подключение к датчикам Tank…')
        def record():
            nonlocal archive,started
            if archive:return
            if not feed.latest or stream_age(feed.latest,'rgb')>2:
                ui.notify('Сначала дождитесь свежих данных',type='warning');return
            archive=SensorArchive();started=time.monotonic();save_state.set_text('Запись датчиков…')
        def stop_record():
            nonlocal archive
            if archive:
                archive.finish();save_state.set_text(f'Сохранено {archive.count} пакетов · запись доступна в списке ниже');archive=None
        if not recording:
            with ui.row().classes('items-center gap-3'):
                ui.button('Подключить датчики',on_click=connect,color='teal')
                ui.button('Записать датчики · до 60 с',on_click=record,color='teal').props('outline')
                ui.button('Закончить запись',on_click=stop_record).props('flat')
        else:
            ui.link('← К живым датчикам','/tank/sensors')
            ui.label(f'Просмотр сохранённой записи · {len(replay_rows)} пакетов').classes('text-lg')
            selector=ui.slider(min=0,max=max(1,len(replay_rows)-1),step=1,value=0).classes('w-full')
        save_state=ui.label('Исходные измерения сохраняются отдельно от показов движения.').classes('muted text-sm')
        if not recording:
            with ui.expansion('Сохранённые записи датчиков').classes('w-full sensor-card'):
                @ui.refreshable
                def recordings():
                    entries=archive_entries()
                    if not entries:ui.label('Пока нет записей')
                    for entry in entries:
                        label=time.strftime('%d.%m %H:%M',time.localtime(entry['created_at']))+f" · {entry['frames']} пакетов"
                        ui.link(label,'/tank/sensors?recording='+entry['id'])
                ui.button('Обновить список',on_click=recordings.refresh).props('flat')
                recordings()
            with ui.expansion('Микрофон · отдельная проверка').classes('w-full sensor-card'):
                ui.label('Обнаружено USB-аудиоустройство. Запись включается только этой кнопкой, на 5 секунд; звук на роботе не воспроизводится.').classes('muted text-sm')
                audio_status=ui.label('Микрофон не записывает')
                playback=ui.audio('').classes('w-full');playback.set_visibility(False)
                async def capture_audio():
                    from .sensors import SensorClient
                    import uuid
                    audio_button.disable();audio_status.set_text('Запись 5 секунд…')
                    def fetch():
                        client=SensorClient()
                        try:
                            result=client.http.get('/microphone.wav',timeout=9);result.raise_for_status();return result.content
                        finally:client.close()
                    try:
                        payload=await run.io_bound(fetch)
                        ident=uuid.uuid4().hex;directory=DATA/'sensor-audio';directory.mkdir(exist_ok=True)
                        (directory/('tank-'+ident+'.wav')).write_bytes(payload)
                        playback.set_source('/api/tank/sensors/audio/'+ident)
                        playback.set_visibility(True)
                        audio_status.set_text('Запись готова. Прослушивание — на этом устройстве.')
                    except Exception as exc:audio_status.set_text('Не удалось записать микрофон: '+str(exc))
                    finally:audio_button.enable()
                audio_button=ui.button('Записать микрофон · 5 с',on_click=capture_audio,color='teal').props('outline')
        with ui.element('div').classes('sensor-grid'):
            images={}
            for key,title,description in (
                ('rgb','Обычная камера','RGB · 640 × 400'),
                ('depth','Расстояния до поверхностей','Глубина в метрах · 0,15–4 м · чёрное = нет измерения'),
                ('scan','Лидар · вид сверху','Кольца через 1 м · верх = +X датчика · обзор до 4,8 м'),
                ('ir','Инфракрасная камера','Отдельный исходный поток Aurora930')):
                with ui.card().classes('sensor-card w-full'):
                    ui.label(title).classes('text-lg font-semibold')
                    ui.label(description).classes('muted text-xs')
                    images[key]=ui.image().props('no-spinner no-transition').classes('w-full').style('aspect-ratio:1.6;background:#10161f')
        with ui.card().classes('sensor-card w-full'):
            ui.label('Поверхности в 3D').classes('text-lg font-semibold')
            ui.label('Облако производителя, в системе камеры: X вправо, Y вниз, Z вперёд. Вращайте мышью; колесо — масштаб.').classes('muted text-sm')
            with ui.scene(width=600,height=420,grid=False,background_color='#10161f').classes('w-full') as scene:
                cloud=scene.point_cloud([[0,0,0]],[[.35,.86,.78]],point_size=.018)
                scene.axes_helper(.3)
                scene.move_camera(x=2,y=-3,z=2,look_at_x=0,look_at_y=0,look_at_z=1)
        columns=[{'name':k,'label':label,'field':k,'align':'left'} for k,label in
                 [('name','Датчик'),('state','Свежесть'),('frame','Система координат'),('details','Измерение')]]
        table=ui.table(columns=columns,rows=[],row_key='name').classes('w-full sensor-card')
        ui.label('Это отдельные измерения робота, не общая карта помещения. Объединение роботов требует проверки положения датчиков и общей системы координат.').classes('muted text-sm')
        with ui.expansion('Как данные используются в обучении').classes('w-full sensor-card'):
            ui.label('Сохраняются исходные глубина, RGB, ИК, скан, IMU, параметры камер и метки времени. Дополнительно рассчитываются 178 признаков окружения: сектора лидара, сетка глубины, IMU, маски пропусков и возраст измерений.').classes('muted')
            ui.label('Запись этого экрана не содержит действий и не является демонстрацией задачи. Обучение здесь не запускается. Потоки асинхронны; их метки времени сохранены для последующего совмещения.').classes('muted')
        async def update():
            nonlocal last_packet,busy,replay_packet,replay_index
            if busy:return
            if recording:
                selected=min(int(selector.value),len(replay_rows)-1)
                if replay_index!=selected:
                    replay_packet=archive_packet(replay_root,replay_rows[selected]);replay_index=selected
                packet=replay_packet
            else:packet=feed.latest
            if packet is None:
                if feed.error:state.set_text('Нет данных: '+feed.error)
                return
            busy=True
            try:
                display_time=packet['received_ns'] if recording else time.monotonic_ns()
                age=stream_age(packet,'rgb',display_time)
                state.set_text(f'Запись · пакет {replay_index+1} из {len(replay_rows)}' if recording else
                    ('Связь потеряна · '+feed.error) if feed.error else ('Данные поступают' if age<2 else 'Данные устарели'))
                rows=[]
                for key,info in packet['meta']['streams'].items():
                    if key.endswith('_info') or key=='tf_static':continue
                    value=stream_age(packet,key,display_time)
                    detail=f"{info['voltage']:.3f} В · на зарядке не означает полный заряд" if key=='battery' else f"{info.get('width','')} × {info.get('height','')}" if 'width' in info else f"{len(packet['arrays'].get(key,[]))} значений"
                    rows.append({'name':key,'state':f'{value:.2f} с' if value<2 else f'Устарело · {value:.1f} с','frame':info.get('frame_id','—'),'details':detail})
                table.rows=rows;table.update()
                if packet is not last_packet:
                    rendered=await run.io_bound(previews,packet)
                    for key,src in rendered.items():images[key].set_source(src)
                    points=packet['arrays'].get('cloud')
                    if points is not None:
                        colors=np.tile([.35,.86,.78],(len(points),1))
                        cloud.set_points(points.tolist(),colors.tolist())
                    if archive:
                        await run.io_bound(archive.add,packet)
                        save_state.set_text(f'Запись: {archive.count} пакетов · {time.monotonic()-started:.0f} с')
                    last_packet=packet
                if archive and time.monotonic()-started>=60:stop_record()
            finally:busy=False
        ui.timer(.5,update)
        ui.context.client.on_disconnect(stop_record)
