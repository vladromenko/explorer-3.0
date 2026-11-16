import json
import time

import pytest

from explorer3 import catalog, datasets, knowledge, paths, training_drafts
from explorer3.cognitive import registry, store


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    for module in (catalog, paths, store):
        monkeypatch.setattr(module, 'DATA', tmp_path)
    monkeypatch.setattr(catalog, 'prepare', lambda: None)
    monkeypatch.setattr(training_drafts, 'checkpoint_hash', lambda path: 'a'*64)
    monkeypatch.setattr(knowledge, 'load_packet', lambda sha: {'donor': {'domain':'sim'}})
    plan = {'domain':'sim', 'sim_model_hash':'b'*64, 'splits': {'train':[], 'validation':[], 'test':[]}}
    monkeypatch.setattr(datasets, 'load_plan', lambda ident: plan)
    catalog.initialize();store.initialize()
    train = {'kind':'train', 'plan_id':'plan', 'donor_packet_shas':['c'*64]}
    config = {'kind':'sim_evaluate', 'source_job':'trained', 'verify_grasp':True, 'episodes':5,
              'profile':'rosmaster_floor_pick_sim', 'expected_model_hash':'b'*64}
    with catalog.connect() as db:
        for ident,cfg in (('trained',train), ('evaluated',config)):
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',
                       (ident,'rosmaster_floor_pick_sim','smolvla','completed',json.dumps(cfg),time.time(),None,'/fixture/weights'))
    report = {'domain':'sim', 'policy':'smolvla', 'independent_grasp_verification':True, 'checkpoint_bundle_sha256':'a'*64,
              'verifier_sha256':registry.file_hash('explorer3/sim/skill_transfer.py'),
              'model':{'xml_sha256':'b'*64}, 'trials':[]}
    for seed in range(5):
        report['trials'].append({'seed':seed, 'outcome':'success', 'donor_online':False, 'geometric_correction':False,
            'object':{'name':'cube', 'geometry':'box_proxy', 'size_m':[.032]*3},
            'grasp_verification':{'accepted':True, 'lift_m':.08, 'held_s':1.5,
                                  'forbidden_contact_pairs':[], 'max_arm_penetration_m':0.}})
    path=tmp_path/'jobs/evaluated/result.json';path.parent.mkdir(parents=True)
    def save():
        path.write_text(json.dumps(report))
    save()
    return report,plan,save,tmp_path


def test_complete_series_is_kept_and_accepted_only_in_measured_sim_scope(evaluation):
    report,_,save,_=evaluation
    report['trials'][-1]['outcome']='failure';save()
    record=registry.register_sim_grasp_evaluation('trained','evaluated')
    assert record['domain']=='sim' and record['verification']['independent_scenes']==4
    assert record['scope']['size_bounds_m']==[[.032,.032]]*3
    item=store.evidence(record['verification']['evidence_ids'],fresh=False)[0]
    assert item.payload['whole_series_trials']==5 and item.payload['whole_series_successes']==4
    assert item.payload['report_ref']=='jobs/evaluated/result.json'
    assert registry.get(record['skill_id']).real_admission=='blocked'


def test_two_failures_cannot_be_hidden_by_three_good_scenes(evaluation):
    report,_,save,_=evaluation
    for trial in report['trials'][:2]:
        trial['outcome']='failure'
    save()
    with pytest.raises(ValueError,match='80 percent'):
        registry.register_sim_grasp_evaluation('trained','evaluated')
    with catalog.connect() as db:
        assert db.execute('SELECT count(*) FROM cognitive_skills').fetchone()[0]==0


def test_contact_failure_and_online_donor_do_not_pass_as_autonomous_success(evaluation):
    report,_,save,_=evaluation
    report['trials'][0]['donor_online']=True
    report['trials'][1]['grasp_verification']['forbidden_contact_pairs']=[['arm','floor']]
    save()
    with pytest.raises(ValueError,match='80 percent'):
        registry.register_sim_grasp_evaluation('trained','evaluated')


def test_reused_recording_seed_and_changed_checkpoint_are_rejected(evaluation,monkeypatch):
    report,plan,save,root=evaluation
    directory=root/'recorded';directory.mkdir()
    (directory/'provenance.json').write_text(json.dumps({'hardware':{'seed':2}}))
    plan['splits']['train']=[{'dataset':str(directory)}]
    with pytest.raises(ValueError,match='overlap'):
        registry.register_sim_grasp_evaluation('trained','evaluated')
    plan['splits']['train']=[]
    monkeypatch.setattr(training_drafts,'checkpoint_hash',lambda path:'d'*64)
    with pytest.raises(ValueError,match='checkpoint'):
        registry.register_sim_grasp_evaluation('trained','evaluated')
