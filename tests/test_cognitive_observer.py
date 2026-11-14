import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from explorer3 import catalog,resources
from explorer3.cognitive import observer,store,experiments
from explorer3.cognitive.schemas import ObservationEvidence
from explorer3.jobs import Jobs


@pytest.fixture
def db(tmp_path,monkeypatch):
    for module in (catalog,resources,store,observer,experiments):monkeypatch.setattr(module,'DATA',tmp_path)
    catalog.initialize();store.initialize();return tmp_path


def frame(db):
    path=db/'rgb.png';Image.new('RGB',(40,30),'red').save(path)
    payload={'archived':True}
    return store.put_evidence(ObservationEvidence(id='camera-frame',robot='rosmaster',domain='real',source='recorded_rgb_frame',
        sample_id='frame-1',frame_id='camera_optical',camera_id='rosmaster/rgb',created_at=10,expires_at=12,
        received_timestamp_ns=100,clock_domain='fixture',status='observed',payload=payload,content_sha256=store.digest(payload),
        artifact_ref='rgb.png',artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_real_frame_hash_domain_and_no_permission(db):
    e=frame(db);request=observer.vlm_request([e.id],question='Что видно?')
    assert request['motion_authorized'] is request['training_authorized'] is False
    assert request['images'][0]['archive'] is True
    assert len(observer.load_images(request))==1
    (db/'rgb.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):observer.load_images(request)


def test_detection_remains_local_hypothesis_no_fabricated_depth(db):
    e=frame(db)
    result={'method':'yoloe','model':{'version':'fixture'},'input_sha256':e.artifact_sha256,
        'image_size':[40,30],'boxes_xyxy':[[1,2,15,20]],'scores':[.8],'prompt':'cube'}
    value=observer.ground_perception(e.id,result)
    assert value.status=='hypothesis' and value.domain=='real'
    assert value.payload['world_xyz'] is value.payload['depth_m'] is None
    assert value.expires_at==e.expires_at
    with pytest.raises(ValueError,match='another frame'):observer.ground_perception(e.id,{**result,'input_sha256':'f'*64})
    with pytest.raises(ValueError,match='outside image'):observer.ground_perception(e.id,{**result,'boxes_xyxy':[[1,2,400,20]]})
    with pytest.raises(ValueError,match='unambiguous'):observer.vlm_request([e.id],question='cube',perception=value.payload)


def test_multimodal_adapter_preserves_pixels_grid_and_attention():
    class Batch(dict):
        def to(self,_):return self
    pixels=np.ones((2,8));grid=np.array([[1,2,2]]);mask=np.array([[1,1,1,0]])
    processor=SimpleNamespace(tokenizer='fixture',apply_chat_template=lambda *a,**k:Batch(
        input_ids=np.array([[1,2,3,0]]),pixel_values=pixels,image_grid_thw=grid,attention_mask=mask))
    model=SimpleNamespace(processor=processor,model=SimpleNamespace(device='cpu'),
        _prepare_completion_kwargs=lambda **k:{'messages':[],'tools':[]},make_stopping_criteria=lambda *a:None)
    values=observer.multimodal_completion(model,[],None,None,{'max_input_tokens':20,'max_new_tokens':10})
    assert values['pixel_values'] is pixels and values['image_grid_thw'] is grid
    assert values['attention_mask'] is mask and values['inputs'].shape==(1,4)


def test_observer_queue_deduplicates_without_launching(db,monkeypatch):
    e=frame(db);jobs=Jobs(None)
    first=jobs.cognitive_observation([e.id],'cube','yoloe')
    assert jobs.cognitive_observation([e.id],'cube','yoloe')==first
    assert jobs.process is None and jobs.thread is None
    cfg=json.loads(jobs.get(first)['config']);assert cfg['kind']=='cognitive_observation'
    assert cfg['real_authorized'] is cfg['training_requested'] is False
    jobs.cancel(first);assert jobs.get(first)['status']=='cancelled'


def test_comparison_freezes_memory_and_equal_choices_no_queue(db):
    ident,plan=experiments.prepare([31,32])
    assert len(plan['cells'])==8 and not catalog.rows('jobs')
    assert plan['memory_updates'] is False and plan['weights_frozen'] is True
    jobs=Jobs(None);first=jobs.cognitive_comparison(ident,'31-M0')
    assert jobs.cognitive_comparison(ident,'31-M0')==first
    assert jobs.process is None and len(catalog.rows('jobs'))==1
    assert experiments.report(ident,[])['complete'] is False
    with pytest.raises(ValueError,match='Unknown comparison cell'):experiments.cell_config(ident,'31-M9')
    path=db/'cognitive/comparisons'/(ident+'.json');path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):experiments.load(ident)


def test_real_preparation_is_shadow_and_cannot_bypass_incident(db,monkeypatch):
    from explorer3.cognitive import real_preparation
    monkeypatch.setattr(real_preparation,'DATA',db)
    e=frame(db);ident,result=real_preparation.prepare('rosmaster_dock',[e.id])
    assert result['mode']=='shadow' and result['motion_authorized'] is False
    assert result['recorded_observations'] and not result['live_observations']
    assert result['problems'] and (db/'cognitive/real-preparations'/(ident+'.json')).is_file()
    with pytest.raises(ValueError,match='another robot'):real_preparation.prepare('tank_dock',[e.id])


def test_assistance_uses_conditions_cost_and_rejects_model_claims(db):
    from explorer3.cognitive.assistance import select
    e=frame(db)
    payload={'conditions':{'recipient_can_continue':True,'observation_skill_available':True},
        'risk':.9,'gain':.4,'gain_status':'available','operator_cost':{'switch_seconds':2,'correction_seconds':8}}
    m=e.model_copy(update={'payload':payload,'source':'vlm'})
    baseline=select(['continue','observe','human'],[m])
    assert baseline['selection']=='continue' and baseline['gain'] is None
    trusted=select(['continue','observe','human'],[m.model_copy(update={'source':'execution_monitor'})])
    assert trusted['selection']=='observe' and trusted['gain_per_operator_second']==pytest.approx(.04)
    refused=select(['recovery'],[m]);assert refused['selection']=='wait' and refused['rejected']
