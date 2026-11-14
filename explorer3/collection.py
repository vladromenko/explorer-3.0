"""Recording series and post-save quality checks; reuses the existing writers/controllers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
from . import catalog
from .paths import DATA

TASKS={
 'so101_pick_place':('SO-101 · кубик в коробку',('so101_pick_place',)),
 'rosmaster_pick_place':('ROSMASTER · кубик в коробку',('rosmaster_pick_place','rosmaster_pick_place_rgbd_lidar')),
 'tank_dock':('Tank · подъезд к цели',('tank_dock_rgbd_lidar','tank_dock')),
 'rosmaster_dock':('ROSMASTER · подъезд к цели',('rosmaster_dock_rgbd_lidar','rosmaster_dock')),
}
_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='recording-readback')


def initialize():
    with catalog.connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS collection_series (
          id TEXT PRIMARY KEY,task TEXT NOT NULL,profile TEXT NOT NULL,layout TEXT NOT NULL,
          operator TEXT NOT NULL,description TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS collection_attempts (
          id TEXT PRIMARY KEY,series TEXT NOT NULL,phase TEXT NOT NULL,outcome TEXT,
          frames INTEGER NOT NULL DEFAULT 0,streams TEXT NOT NULL DEFAULT '{}',duration REAL,
          episode TEXT,quality TEXT,user_excluded INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL);
        ''')


def series(task=None):
    initialize()
    with catalog.connect() as db:
        return [dict(r) for r in db.execute('SELECT * FROM collection_series'+(' WHERE task=?' if task else '')+' ORDER BY created',
                                           (task,) if task else ())]


def create_series(task,profile,operator,description,repeat_layout=None):
    if task not in TASKS or profile not in TASKS[task][1]:raise ValueError('Выберите вариант этой задачи')
    operator=operator.strip();description=description.strip()
    if not operator or len(operator)>80:raise ValueError('Укажите оператора (до 80 знаков)')
    if not description or len(description)>500:raise ValueError('Кратко опишите расстановку (до 500 знаков)')
    initialize()
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        existing=[dict(r) for r in db.execute('SELECT * FROM collection_series WHERE task=? ORDER BY created',(task,))]
        if existing and existing[0]['profile']!=profile:
            raise ValueError('В этой задаче уже выбран другой вариант; не смешивайте схемы данных')
        number=len(existing)+1
        ident=f'{task}-s{number:02}'
        if repeat_layout:
            if repeat_layout not in {r['layout'] for r in existing}:raise ValueError('Такой прежней расстановки нет в этой задаче')
            layout=repeat_layout
        else:layout=f'{task}-layout{number:02}'
        db.execute('INSERT INTO collection_series VALUES(?,?,?,?,?,?,?)',
                   (ident,task,profile,layout,operator,description,time.time()))
    return next(r for r in series(task) if r['id']==ident)


def context(profile,session,scene):
    # Ordinary legacy recordings remain valid; only managed series have a fixed profile/layout.
    initialize()
    with catalog.connect() as db:row=db.execute('SELECT * FROM collection_series WHERE id=?',(session,)).fetchone()
    if row is None:return None
    value=dict(row)
    if value['profile']!=profile or value['layout']!=scene:
        raise ValueError('Профиль или расстановка не совпадают с выбранной серией')
    return value


def begin(ident,series_id):
    with catalog.connect() as db:
        db.execute('INSERT INTO collection_attempts(id,series,phase,created) VALUES(?,?,?,?)',
                   (ident,series_id,'preparing',time.time()))


def progress(ident,frames,streams):
    with catalog.connect() as db:
        db.execute('UPDATE collection_attempts SET phase=?,frames=?,streams=? WHERE id=?',
                   ('recording',frames,json.dumps(streams),ident))


def completed(ident,outcome,frames,episode=None,reason=None):
    with catalog.connect() as db:
        db.execute('UPDATE collection_attempts SET phase=?,outcome=?,frames=?,episode=?,quality=? WHERE id=?',
            ('checking' if episode else 'saved','aborted' if not frames else outcome,frames,episode,
             json.dumps({'status':'warning','problems':[reason or 'Запись прервана до первого кадра']}),ident))
    if episode:
        catalog.exclude(episode,True) # pending readback is never a training example
        _pool.submit(audit,episode)


def audit(episode):
    problems=[];duration=None;thumbnails=[]
    try:
        entry=next(e for e in catalog.rows('episodes') if e['id']==episode)
        root=Path(entry['dataset']);provenance=json.loads((root/'provenance.json').read_text())
        from .validation import validate
        result=validate(episode,domain=provenance.get('domain','real'))
        duration=result['duration_seconds']
        if not 3<=duration<=120:problems.append('Длительность должна быть от 3 до 120 секунд')
        rows=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
        stamps=[r['observation_timestamp_ns'] for r in rows]
        if len(stamps)>1 and max(np.diff(stamps))>200_000_000:problems.append('Между наблюдениями есть пропуск больше 200 мс')
        cameras=[r['camera'].get('agent_receive_timestamp_ns',r['camera']['timestamp_ns']) for r in rows]
        unique=sorted(set(cameras))
        if len(unique)<2 or max(np.diff(unique),default=0)>200_000_000:
            problems.append('В камере нет непрерывного потока с пропусками не более 200 мс')
        # Check repeated/frozen camera stamps against observation time, not only unique differences.
        last=None;since=stamps[0]
        for stamp,camera in zip(stamps,cameras):
            if camera!=last:last=camera;since=stamp
            if stamp-since>200_000_000:
                problems.append('Кадр камеры повторялся дольше 200 мс');break
        if provenance['profile'].get('environment_channels'):
            from .sensors import decode_packet
            seen={};ages={}
            for ref in sorted({r['sensor_packet'] for r in rows}):
                packet=decode_packet((root/ref).read_bytes());meta=packet['meta']
                for name in ('depth','scan','imu')+ (('scan_rear',) if provenance['profile']['robot']=='rosmaster' else ()):
                    stream=meta['streams'].get(name,{})
                    if stream.get('received_ns'):seen.setdefault(name,set()).add(stream['received_ns'])
                    ages[name]=max(ages.get(name,0),stream.get('age_s',float('inf')))
                    history=packet['arrays'].get(name+'_history_received_ns')
                    if history is not None:seen.setdefault(name,set()).update(map(int,history))
            for name,age in ages.items():
                values=sorted(seen.get(name,()))
                if age>.2 or len(values)<2 or max(np.diff(values),default=0)>200_000_000:
                    problems.append(name+': есть пропуски больше 200 мс или недостаточно временных меток')
        # Full numeric action/state alignment, not just three decoded video frames.
        import pyarrow.parquet as pq
        files=sorted((root/'data').rglob('*.parquet'))
        numeric=[row for f in files for row in pq.read_table(f).to_pylist()]
        if len(numeric)!=len(rows) or any(not np.allclose(a['action'],b['action'],atol=1e-5) or
                not np.allclose(a['observation.state'],b['observation.state'],atol=1e-5) for a,b in zip(numeric,rows)):
            problems.append('Действия или состояния в датасете не совпадают с журналом')
        import av
        from PIL import Image
        folder=DATA/'validation/collection'/episode;folder.mkdir(parents=True,exist_ok=True)
        first=last=None
        for file in sorted((root/'videos').rglob('*.mp4')):
            with av.open(str(file)) as video:
                for frame in video.decode(video=0):
                    if first is None:first=frame.to_image()
                    last=frame.to_image()
        for name,picture in (('first',first),('last',last)):
            if picture is not None:
                picture.thumbnail((480,360));picture.save(folder/(name+'.jpg'));thumbnails.append(name)
        quality={'status':'warning' if problems else 'pass','problems':problems,'readback':result,
                 'thumbnails':thumbnails,'duration_s':duration,'checked_at':time.time(),
                 'size_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    except Exception as exc:
        quality={'status':'warning','problems':['Не удалось проверить запись: '+str(exc)],'checked_at':time.time()}
    destination=DATA/'validation/collection'/episode;destination.mkdir(parents=True,exist_ok=True)
    from .fleet_config import atomic_write
    atomic_write(destination/'quality.json',quality)
    with catalog.connect() as db:
        db.execute('UPDATE collection_attempts SET phase=?,duration=?,quality=? WHERE episode=?',
                   ('saved',duration,json.dumps(quality,ensure_ascii=False),episode))
        row=db.execute('SELECT user_excluded FROM collection_attempts WHERE episode=?',(episode,)).fetchone()
        excluded=quality['status']!='pass' or quality.get('readback',{}).get('synthetic',False) or row is None or bool(row[0])
    catalog.exclude(episode,excluded)
    return quality


def exclude(episode):
    with catalog.connect() as db:db.execute('UPDATE collection_attempts SET user_excluded=1 WHERE episode=?',(episode,))
    catalog.exclude(episode,True)


def summary(series_id):
    initialize()
    with catalog.connect() as db:
        selected=db.execute('SELECT * FROM collection_series WHERE id=?',(series_id,)).fetchone()
        if selected is None:raise ValueError('Выберите серию')
        selected=dict(selected)
        entries=[dict(r) for r in db.execute('SELECT a.*,s.layout FROM collection_attempts a JOIN collection_series s ON a.series=s.id WHERE s.task=? ORDER BY a.created',(selected['task'],))]
        groups=db.execute('SELECT COUNT(DISTINCT layout) FROM collection_series WHERE task=?',(selected['task'],)).fetchone()[0]
    current=[r for r in entries if r['series']==series_id]
    counts=lambda rows:{k:sum(r['outcome']==k for r in rows) for k in ('success','failure','aborted')}
    durations=[r['duration'] for r in entries if r['duration'] is not None]
    total=counts(entries)
    accepted=[r for r in entries if r['outcome']=='success' and not r['user_excluded'] and json.loads(r['quality'] or '{}').get('status')=='pass']
    remaining=max(0,50-len(accepted))
    sizes=[json.loads(r['quality'] or '{}').get('size_bytes',0) for r in entries]
    sizes=[s for s in sizes if s>0]
    return {'series':selected,'current':counts(current),'total':total,'to_50':remaining,
            'independent_groups':groups,'groups_with_accepted_data':len({r['layout'] for r in accepted}),'usable_success':len(accepted),'target':'10 серий × 5 успешных',
            'mean_duration_s':sum(durations)/len(durations) if durations else None,
            'estimated_remaining_bytes':int(sum(sizes)/len(sizes)*remaining) if sizes else None,
            'attempts':current,'free_bytes':shutil.disk_usage(DATA).free}
