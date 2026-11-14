"""Upstream ByteTrack over saved detections; IDs are local to one camera/run."""
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from explorer3 import catalog
from explorer3.paths import DATA
from . import observer,store


def prepare(job_ids):
    if not 2<=len(job_ids)<=32 or len(set(job_ids))!=len(job_ids):raise ValueError('Нужны 2–32 разных разбора последовательных кадров')
    jobs={j['id']:j for j in catalog.rows('jobs')};frames=[]
    for ident in job_ids:
        row=jobs.get(ident)
        if row is None or row['status']!='completed':raise ValueError('Нужны завершённые разборы YOLOE')
        config=json.loads(row['config'])
        if config.get('kind')!='cognitive_observation' or config.get('method')!='yoloe':raise ValueError('Нужен YOLOE с исходным evidence')
        result=json.loads((DATA/'jobs'/ident/'result.json').read_text())
        spec=observer.image_input(config['request']['evidence_ids'][0])
        if result.get('input_sha256')!=spec['sha256']:raise ValueError('Detection frame changed')
        frames.append({'job_id':ident,'result_sha256':store.digest(result),'source':spec,
            'prompt':result['prompt'],'model':result['model'],'image_size':result['image_size'],
            'boxes_xyxy':result['boxes_xyxy'],'scores':result['scores']})
    frames.sort(key=lambda x:x['source']['created_at'])
    if len({(f['source']['robot'],f['source']['camera_id'],f['source']['domain'],f['prompt'],store.digest(f['model'])) for f in frames})!=1:
        raise ValueError('Одна камера, один объект и одна версия детектора на последовательность')
    times=[f['source']['created_at'] for f in frames]
    gaps=np.diff(times)
    if not np.isfinite(gaps).all() or (gaps<=0).any() or (gaps>1.).any():raise ValueError('Кадры должны идти по порядку с интервалом не больше секунды')
    return {'kind':'cognitive_tracking','frames':frames,'device':'cpu','fps':float(1/np.median(gaps)),
        'training_requested':False,'physical_access':False,'tracking_scope':'one camera sequence, not global identity'}


def associate(frames,fps,scope):
    if version('ultralytics')!='8.4.14' or version('lap')!='0.5.12':raise ValueError('Pinned tracker runtime changed')
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes
    tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
        track_buffer=30,match_thresh=.8,fuse_score=True),frame_rate=max(1,round(fps)))
    output=[]
    for frame in frames:
        boxes=frame['boxes_xyxy'];scores=frame['scores'];width,height=frame['image_size']
        if len(boxes)!=len(scores):raise ValueError('Detection count mismatch')
        rows=[]
        for box,score in zip(boxes,scores):
            if len(box)!=4 or not all(math.isfinite(v) for v in [*box,score]):raise ValueError('Invalid detection')
            if not 0<=box[0]<=box[2]<=width or not 0<=box[1]<=box[3]<=height or not 0<=score<=1:raise ValueError('Detection outside frame')
            rows.append([*box,score,0])
        detections=Boxes(np.asarray(rows,dtype=np.float32).reshape(-1,6),(height,width))
        tracks=tracker.update(detections)
        output.append({'evidence_id':frame['source']['evidence_id'],
            'tracks':[{'id':scope+':'+str(int(t[4])),'box_xyxy_px':[float(x) for x in t[:4]],'score':float(t[5])} for t in tracks]})
    return output


def execute(config,work):
    current=prepare([f['job_id'] for f in config['frames']])
    if current!=config:raise ValueError('Tracking inputs changed')
    frames=config['frames'];camera=frames[0]['source']['robot']+'/'+frames[0]['source']['camera_id']
    result={'tracker':'ultralytics.ByteTrack','ultralytics':'8.4.14','lap':'0.5.12',
        'frames':associate(frames,config['fps'],camera+':'+work.name),
        'source_jobs':[f['job_id'] for f in frames],'inputs_sha256':store.digest(config),
        'domain':frames[0]['source']['domain'],'status':'hypothesis','world_xyz':None,
        'global_object_identity':False,'model_weights_loaded':False,'training_started':False,
        'physical_access':False,'task_outcome':'unknown'}
    (work/'result.json').write_text(store.encoded(result));return result


if __name__=='__main__':execute(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
