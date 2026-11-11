"""Read-only episode playback. No robot imports or command execution."""
import json
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app,ui
from . import catalog
from .paths import DATA

def episode_files(ident: str):
    rows=[e for e in catalog.rows('episodes') if e['id']==ident]
    if len(rows)!=1:
        raise HTTPException(404,'Episode not found')
    root=Path(rows[0]['dataset']).resolve()
    if not root.is_relative_to(DATA/'datasets'):
        raise HTTPException(403,'Dataset outside Explorer storage')
    return rows[0],root

@app.get('/api/episodes/{ident}/video')
def episode_video(ident: str):
    _,root=episode_files(ident)
    cfg=json.loads((root/'provenance.json').read_text())['profile']
    videos=sorted((root/('videos/observation.images.'+cfg['camera'])).rglob('*.mp4'))
    if len(videos)!=1:
        raise HTTPException(409,'Expected one finalized camera video file')
    return FileResponse(videos[0],media_type='video/mp4')

@ui.page('/episodes/{ident}')
def episode_page(ident: str):
    row,root=episode_files(ident)
    samples=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
    profile=json.loads((root/'provenance.json').read_text())['profile']
    ui.dark_mode().enable()
    ui.add_css('body{background:#10161f;color:#e6edf4;font-family:system-ui}.nicegui-content{max-width:1100px;margin:auto;padding:24px}.q-card{background:#192331}')
    ui.link('← Explorer 3.0','/')
    ui.label('Просмотр эпизода').style('font-size:28px;font-weight:600')
    ui.label(f"{row['profile']} · {row['outcome']} · {row['frames']} кадров")
    ui.label('Только просмотр. Команды на робота не передаются.')
    video=ui.video(f'/api/episodes/{ident}/video').style('width:100%;max-width:800px')
    channel=ui.select({i:n for i,n in enumerate(profile['channels'])},value=0,label='Канал').style('min-width:260px')
    state_label='Измеренное положение' if profile['robot']=='so101' else 'Предыдущая команда (оценка)'
    plot=ui.echart({'backgroundColor':'transparent','tooltip':{'trigger':'axis'},
        'legend':{'data':['Отправленная команда',state_label],'textStyle':{'color':'#b8c6d6'}},
        'xAxis':{'type':'value','name':'Видео, с'},'yAxis':{'type':'value'},
        'series':[{'name':label,'type':'line','showSymbol':False,'data':[]} for label in ('Отправленная команда',state_label)]}).style('width:100%;height:320px')
    clock=ui.label('0.0 с')
    def draw():
        index=channel.value
        stride=max(1,len(samples)//1500)
        for series,key in zip(plot.options['series'],('action','observation.state')):
            series['data']=[[s['frame_index']/profile['fps'],s[key][index]] for s in samples[::stride]]
        plot.options['yAxis']['name']=profile['units'][index]
        plot.update()
    def timestamp(event):
        seconds=float(event.args)
        clock.set_text(f'{seconds:.1f} с')
        plot.options['series'][0]['markLine']={'symbol':'none','silent':True,'data':[{'xAxis':seconds}],
                                            'lineStyle':{'color':'#eecc65'},'label':{'show':False}}
        plot.update()
    video.on('timeupdate',timestamp,js_handler='(event) => emit(event.target.currentTime)',throttle=.25)
    channel.on_value_change(draw)
    draw()
