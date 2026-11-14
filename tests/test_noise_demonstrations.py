"""Numeric and file-contract checks only; no simulator, policy or trainer."""
import io
import json
from types import SimpleNamespace
import numpy as np
import pytest
from explorer3 import noise_demonstrations as noise


def test_noise_scale_is_measured_bounded_reproducible_and_intention_unchanged():
    intent=[.99,-.99];rng=np.random.default_rng(24)
    noisy,sigma=noise.perturb(intent,[10,.02],[-1,-1],[1,1],rng,noisy=True)
    assert sigma.tolist()==[.1,.02] and intent==[.99,-.99]
    assert np.all(noisy<=1) and np.all(noisy>=-1)
    repeated,_=noise.perturb(intent,[10,.02],[-1,-1],[1,1],np.random.default_rng(24),noisy=True)
    assert np.array_equal(noisy,repeated)
    clean,_=noise.perturb(intent,[10,.02],[-1,-1],[1,1],rng,noisy=False)
    assert clean.tolist()==intent
    with pytest.raises(ValueError):noise.perturb([2],[.1],[-1],[1],rng,noisy=True)


def test_expert_target_never_relabels_real_commands():
    sample={'action':[.3,.2],'expert_action':[.1,.1],'command_source':'scripted_expert'}
    assert noise.training_target(sample,domain='sim',label='expert_intent_before_noise')==[.1,.1]
    assert noise.training_target(sample,domain='real',label=None)==[.3,.2]
    with pytest.raises(ValueError):noise.training_target(sample,domain='real',label='expert_intent_before_noise')


def test_writer_keeps_actual_command_sidecar_and_separate_training_target():
    from explorer3.recording import EpisodeWriter
    writer=EpisodeWriter.__new__(EpisodeWriter);frames=[]
    writer.camera='front';writer.domain='sim';writer.category='correction';writer.command_source=None
    writer.training_label='expert_intent_before_noise';writer.initial_state=None;writer.last_observation=-1
    writer.cfg={'channels':['v','w'],'task':'fixture'};writer.ds=SimpleNamespace(add_frame=frames.append)
    writer.id='fixture';writer.robot_id='tank';writer.name='tank_dock_sim';writer.session='fixture';writer.scene='fixture'
    writer.frames=0;writer.sidecar=io.StringIO()
    sample={'observation.state':[0,0],'action':[.2,.1],'expert_action':[.1,0],'command_source':'scripted_expert',
        'observation_timestamp_ns':1,'action_send_start_ns':1,'action_send_end_ns':1,
        'images':{'front':{'image':np.zeros((8,8,3),np.uint8),'timestamp_ns':1}}}
    writer.add(sample)
    assert np.allclose(frames[0]['action'],[.1,0])
    record=json.loads(writer.sidecar.getvalue())
    assert record['action']==[.2,.1] and record['expert_action']==[.1,0]


def test_prepare_requires_real_validation_not_smoke_or_test(monkeypatch,tmp_path):
    monkeypatch.setattr(noise,'DATA',tmp_path)
    row={'id':'source','status':'completed','policy':'act','checkpoint':str(tmp_path/'weights'),
        'profile':'tank_dock_sim','config':json.dumps({'smoke':True})}
    monkeypatch.setattr(noise.catalog,'rows',lambda _: [row])
    with pytest.raises(ValueError,match='validation'):noise.prepare('source',[99])
    row['config']=json.dumps({'smoke':False,'plan_id':'fixture'})
    monkeypatch.setattr(noise.datasets,'load_plan',lambda _: {'splits':{},'sim_model_hash':'a'*64})
    folder=tmp_path/'jobs/source';folder.mkdir(parents=True)
    (folder/'evaluation.json').write_text(json.dumps({'role':'held_out_test','checkpoint':row['checkpoint']}))
    with pytest.raises(ValueError,match='validation'):noise.prepare('source',[99])


def test_prepared_collection_pins_measured_error_weights_and_new_seeds(monkeypatch,tmp_path):
    monkeypatch.setattr(noise,'DATA',tmp_path)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}');(checkpoint/'model.safetensors').write_bytes(b'fixture only, not a model')
    entry=tmp_path/'episode';entry.mkdir();(entry/'provenance.json').write_text(json.dumps({'hardware':{'seed':7}}))
    plan={'splits':{'train':[],'validation':[{'dataset':str(entry)}],'test':[]},'sim_model_hash':'a'*64}
    monkeypatch.setattr(noise.datasets,'load_plan',lambda _:plan)
    row={'id':'source','status':'completed','policy':'act','checkpoint':str(checkpoint),'profile':'tank_dock_sim',
        'config':json.dumps({'smoke':False,'plan_id':'fixture'})}
    monkeypatch.setattr(noise.catalog,'rows',lambda _:[row])
    report={'role':'held_out_validation','checkpoint':str(checkpoint),'checkpoint_bundle_sha256':noise.checkpoint_hash(checkpoint),
        'datasets':[{'dataset':str(entry),'samples':4,'rmse_by_channel':[.1,.2],'channels':noise.profile('tank_dock_sim')['channels']}]}
    folder=tmp_path/'jobs/source';folder.mkdir(parents=True);(folder/'evaluation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='seeds'):noise.prepare('source',[7])
    ident,value=noise.prepare('source',[100,101,102]);assert noise.load(ident)==value
    assert value['rmse_by_channel']==[.1,.2] and not value['training_started']
    assert not list((tmp_path/'jobs').glob('*.json'))
    (checkpoint/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):noise.load(ident)
