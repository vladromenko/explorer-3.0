"""Ground perception in immutable robot frames; no model or hardware is opened here."""
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

from PIL import Image
from explorer3.paths import DATA
from . import store
from .schemas import ObservationEvidence


def frame_from_episode(episode_id,frame_index):
    """Decode one original video frame for SO-101 or either mobile platform."""
    import av
    from explorer3 import catalog
    from explorer3.datasets import fingerprint
    row=next((r for r in catalog.rows('episodes') if r['id']==episode_id),None)
    if row is None:raise ValueError('Unknown episode')
    root=Path(row['dataset']).resolve()
    if not root.is_relative_to(DATA.resolve()):raise ValueError('Episode outside Explorer storage')
    meta=json.loads((root/'provenance.json').read_text());cfg=meta['profile']
    if meta.get('domain','real')!='real' or meta.get('synthetic') is not False:raise ValueError('Original real recording required')
    if type(frame_index) is not int or not 0<=frame_index<row['frames']:raise ValueError('Frame outside episode')
    videos=list((root/('videos/observation.images.'+cfg['camera'])).rglob('*.mp4'))
    if len(videos)!=1:raise ValueError('One finalized video required')
    rgb=None
    with av.open(videos[0]) as video:
        for i,frame in enumerate(video.decode(video=0)):
            if i==frame_index:rgb=frame.to_image();break
    if rgb is None:raise ValueError('Frame missing from video')
    target=DATA/'cognitive/frames';target.mkdir(parents=True,exist_ok=True)
    path=target/(uuid.uuid4().hex+'.png');rgb.save(path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();fixed=target/(sha+'.png')
    if fixed.exists():path.unlink()
    else:path.replace(fixed)
    lines=(root/'research.jsonl').read_text().splitlines();sample=json.loads(lines[frame_index])
    created=row['created']+frame_index/cfg['fps']
    payload={'archived':True,'episode_id':episode_id,'frame_index':frame_index,'dataset_sha256':fingerprint(root),
        'original_video':str(videos[0].relative_to(DATA)),'original_camera_metadata':sample.get('camera'),
        'time_basis':'episode creation plus frame index; not camera exposure time','world_xyz':None}
    item=ObservationEvidence(id=uuid.uuid4().hex,robot=cfg['robot'],domain='real',source='recorded_episode_frame',
        sample_id=episode_id+':'+str(frame_index),frame_id=cfg['robot']+'/'+cfg['camera'],camera_id=cfg['camera'],
        received_timestamp_ns=sample['observation_timestamp_ns'],clock_domain='homebrain_monotonic',
        created_at=created,expires_at=created+2,status='observed',payload=payload,content_sha256=store.digest(payload),
        artifact_ref=str(fixed.relative_to(DATA)),artifact_sha256=sha)
    return store.put_evidence(item)


def image_input(evidence_id):
    item=store.evidence([evidence_id],fresh=False)[0]
    store.validate_evidence_artifact(item)
    if not item.camera_id or not item.artifact_ref:
        raise ValueError('Evidence has no original camera frame')
    path=(DATA/item.artifact_ref).resolve()
    with Image.open(path) as frame:
        if frame.width*frame.height>4096*4096:raise ValueError('Camera frame is too large')
        frame.verify()
    return {'evidence_id':item.id,'artifact_ref':item.artifact_ref,
        'sha256':item.artifact_sha256,'robot':item.robot,'camera_id':item.camera_id,
        'domain':item.domain,'frame_id':item.frame_id,'sample_id':item.sample_id,
        'created_at':item.created_at,'expires_at':item.expires_at,
        'archive':bool(item.payload.get('archived'))}


def load_images(request,maximum=2):
    """Model subprocess revalidates bytes, identities and image budget before use."""
    specs=request.get('images',[])
    if len(specs)>maximum:raise ValueError('Image budget exceeded')
    frames=[]
    for spec in specs:
        actual=image_input(spec['evidence_id'])
        if spec!=actual:raise ValueError('Image evidence changed after preparation')
        with Image.open(DATA/actual['artifact_ref']) as frame:
            rgb=frame.convert('RGB');rgb.thumbnail((448,448));frames.append(rgb.copy())
    return frames


def frame_from_packet(evidence_id):
    """Decode the recorded RGB only. Its original age/domain are not renewed."""
    from explorer3.sensors import decode_packet
    packet_e=store.evidence([evidence_id],domain='real',fresh=False)[0]
    store.validate_evidence_artifact(packet_e)
    if packet_e.source not in ('recorded_sensor_snapshot','live_sensor_snapshot'):raise ValueError('Original sensor packet required')
    packet=decode_packet((DATA/packet_e.artifact_ref).read_bytes())
    stream=packet['meta']['streams'].get('rgb',{});rgb=packet['arrays'].get('rgb')
    if rgb is None or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('No RGB in this packet')
    root=DATA/'cognitive/frames';root.mkdir(parents=True,exist_ok=True)
    path=root/(uuid.uuid4().hex+'.png')
    Image.fromarray(rgb).save(path)
    age=stream.get('age_s')
    valid_age=isinstance(age,(float,int)) and not isinstance(age,bool) and math.isfinite(age) and age>=0
    created=packet_e.created_at-age if valid_age else packet_e.created_at
    archived=packet_e.payload.get('archived',True)
    if not archived and not valid_age:raise ValueError('Live RGB age is unknown')
    payload={'archived':archived,'packet_evidence_id':packet_e.id,'reported_capture_age_s':age if valid_age else None,
        'world_xyz':None,'rgb_depth_registered':False,'image_size':[rgb.shape[1],rgb.shape[0]]}
    item=ObservationEvidence(id=uuid.uuid4().hex,domain='real',robot=packet_e.robot,source='recorded_rgb_frame' if archived else 'live_rgb_frame',
        sample_id=packet_e.sample_id,frame_id=stream.get('frame_id') or packet_e.frame_id,
        camera_id=packet_e.robot+'/rgb',source_timestamp_ns=stream.get('source_timestamp_ns'),
        received_timestamp_ns=packet_e.received_timestamp_ns,clock_domain=packet_e.clock_domain,
        boot_id=packet_e.boot_id,created_at=created,expires_at=created+2,status='observed',
        artifact_ref=str(path.relative_to(DATA)),artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        content_sha256=store.digest(payload),payload=payload)
    # Use a content-addressed file: importing the same packet cannot overwrite
    # bytes referenced by a previous evidence entry with a different encoder.
    fixed=root/(item.artifact_sha256+'.png')
    if fixed.exists():path.unlink()
    else:path.replace(fixed)
    item=item.model_copy(update={'artifact_ref':str(fixed.relative_to(DATA))})
    return store.put_evidence(item)


def capture_readonly(robot):
    """One existing read-only sensor request; no owner transfer or motor endpoint."""
    from explorer3.sensors import SensorClient,SensorArchive
    from .evidence import import_sensor_packet
    client=SensorClient(isolated=False,robot=robot)
    try:packet=client.snapshot()
    finally:client.close()
    received=time.time()
    archive=SensorArchive(robot=robot)
    try:archive.add(packet)
    finally:archive.finish()
    old=import_sensor_packet(str((archive.root/'000000.npz').relative_to(DATA)))
    # Reception time is Homebrain wall time, with the entire request duration
    # subtracted as a conservative age bound. Not a hardware-clock calibration.
    created=received-packet['request_elapsed_s']
    payload={**old.payload,'archived':False,'request_elapsed_s':packet['request_elapsed_s'],
        'time_basis':'homebrain reception minus request duration; per-stream ages still apply'}
    current=old.model_copy(update={'id':uuid.uuid4().hex,'source':'live_sensor_snapshot',
        'created_at':created,'expires_at':created+2,'payload':payload,'content_sha256':store.digest(payload)})
    store.put_evidence(current)
    return frame_from_packet(current.id)


def ground_perception(evidence_id,result):
    """Model boxes remain hypotheses in the image frame, never world coordinates."""
    spec=image_input(evidence_id)
    if result.get('input_sha256')!=spec['sha256']:raise ValueError('Perception used another frame')
    if result.get('method') not in ('yoloe','sam3'):raise ValueError('Unsupported perception source')
    width,height=result['image_size'];boxes=result['boxes_xyxy'];scores=result['scores']
    if len(boxes)!=len(scores):raise ValueError('Mismatched detection scores')
    detections=[]
    for box,score in zip(boxes,scores):
        if len(box)!=4 or not all(math.isfinite(x) for x in [*box,score]):raise ValueError('Invalid detection')
        x1,y1,x2,y2=box
        if not (0<=x1<=x2<=width and 0<=y1<=y2<=height and 0<=score<=1):raise ValueError('Detection outside image')
        detections.append({'box_xyxy_px':box,'score':score,'label':result['prompt']})
    payload={'detections':detections,'model':result['model'],'source_evidence_id':evidence_id,
        'archived':spec['archive'],'image_sha256':spec['sha256'],'world_xyz':None,
        'depth_m':None,'depth_reason':'No verified RGB-depth registration and timestamp pairing',
        'ambiguity':len(detections)!=1,'task_outcome':'unknown'}
    source=store.evidence([evidence_id],fresh=False)[0]
    return store.put_evidence(source.model_copy(update={'id':uuid.uuid4().hex,
        'source':'grounded_'+result['method'],'status':'hypothesis','payload':payload,
        'content_sha256':store.digest(payload)}))


def vlm_request(evidence_ids,*,question,perception=None):
    if not 1<=len(evidence_ids)<=2:raise ValueError('One or two original frames required')
    if not isinstance(question,str) or not 1<=len(question)<=400:raise ValueError('Short observation question required')
    specs=[image_input(ident) for ident in evidence_ids]
    if len({s['domain'] for s in specs})!=1:raise ValueError('Do not mix real and simulated evidence')
    if perception is not None and not perception.get('ambiguity',True):
        raise ValueError('Use the unambiguous detector result; no VLM fallback needed')
    from .registry import file_hash
    return {'purpose':'scene_observation','question':question,'images':specs,
        'model_config_sha256':file_hash('configs/cognitive-model.json'),
        'evidence_ids':evidence_ids,'perception':perception,
        'instruction':'Describe only visible evidence. Missing/ambiguous information is unknown. '
            'No world coordinates, actuator commands or claims of task success. '
            'These may be old recorded frames; they do not establish current robot state.',
        'prepared_at':time.time(),'training_authorized':False,'motion_authorized':False}


def multimodal_completion(model,messages,stop_sequences,tools_to_call_from,config,**kwargs):
    """Keep processor pixel tensors that smolagents 1.26's text adapter discards."""
    values=model._prepare_completion_kwargs(messages=messages,stop_sequences=stop_sequences,
        tools_to_call_from=tools_to_call_from,tool_choice=None,convert_images_to_image_urls=True,**kwargs)
    messages=values.pop('messages');stops=values.pop('stop',None);tools=values.pop('tools',None)
    batch=model.processor.apply_chat_template(messages,tools=tools,return_tensors='pt',
        add_generation_prompt=True,tokenize=True,return_dict=True).to(model.model.device)
    tensors=dict(batch);tokens=tensors.pop('input_ids')
    if tokens.shape[1]>config['max_input_tokens']:raise ValueError('Input token budget exceeded')
    return {**values,**tensors,'inputs':tokens,'use_cache':True,'max_time':25.,
        'max_new_tokens':config['max_new_tokens'],
        'stopping_criteria':model.make_stopping_criteria(stops,model.processor.tokenizer) if stops else None}
