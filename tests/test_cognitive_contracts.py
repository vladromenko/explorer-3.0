import json
import time
import uuid
import pytest
from explorer3 import catalog,resources
from explorer3.cognitive import store,broker,registry
from explorer3.cognitive.schemas import ObservationEvidence,AssistanceCase,CompatibilityRecord,TaskGraph


@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(catalog,'DATA',tmp_path);monkeypatch.setattr(store,'DATA',tmp_path);monkeypatch.setattr(resources,'DATA',tmp_path)
    catalog.initialize();store.initialize();return tmp_path


def evidence(db,*,identifier=None,domain='sim',source='fixture',ttl=60,payload=None):
    now=time.time();payload={'conditions':{'recipient_stationary':True,'fixed_scene':True}} if payload is None else payload
    e=ObservationEvidence(id=identifier or uuid.uuid4().hex,domain=domain,robot='tank',source=source,sample_id='fixture-frame',
        frame_id='tank/front_cam',received_timestamp_ns=time.time_ns(),clock_domain='fixture_clock',
        created_at=now-1,expires_at=now+ttl,status='observed',content_sha256=store.digest(payload),payload=payload)
    return store.put_evidence(e)


def mission(db):
    e=evidence(db);return broker.create_mission('Подъехать к фиксированной цели',[e.id])


def call(m,tool,args=None,role='MissionCoordinator',identifier=None):
    return {'request_id':identifier or uuid.uuid4().hex,'mission_id':m['id'],'generation':m['generation'],
        'decision_id':m['decision_id'],'role':role,'tool':tool,'mode':'sim','snapshot_ids':m['snapshot_ids'],
        'expires_at':time.time()+30,'arguments':args or {}}


def graph(m):
    c=registry.get('tank.dock_fixed.scripted.v1')
    return {'nodes':[{'id':'dock','skill_id':c.skill_id,'skill_version':c.version,'robot':'tank',
        'goal_id':'fixed_dock','deadline_s':20,'depends_on':[]}],'evidence_ids':m['snapshot_ids'],'rationale':'fixture'}


def candidate(db):
    e=evidence(db);c=registry.get('tank.dock_fixed.scripted.v1')
    scope=CompatibilityRecord(recipient_skill_id=c.skill_id,recipient_version=c.version,policy_sha256=c.policy_sha256,
        camera_sha256=c.camera_sha256,calibration_sha256=None,domain='sim',goal_id='fixed_dock',
        checked_conditions=['fixed_scene'],unknown_conditions=[],evidence_ids=[e.id])
    return AssistanceCase(id=uuid.uuid4().hex,task='dock',effect='target_visible',domain='sim',donor='rosmaster',recipient='tank',
        donor_episode='donor-fixture',recipient_episode='recipient-fixture',reason='Independent camera observation',
        before_evidence_ids=[e.id],compatibility=scope,assistance_type='observation',expected_effect='Verify visible target',
        actions_provenance='observation',actual_effect={})


def test_idempotence_unknown_skill_stale_snapshot_and_stop(db):
    m=mission(db);request=call(m,'propose_graph',graph(m))
    result=broker.call_tool(request);assert result['result']['accepted']
    assert broker.call_tool(request)['replayed']
    changed={**request,'arguments':{}};
    with pytest.raises(ValueError,match='different inputs'):broker.call_tool(changed)
    invalid=graph(m);invalid['nodes'][0]['skill_id']='tank.grasp'
    with pytest.raises(ValueError,match='Unknown skill'):broker.call_tool(call(m,'propose_graph',invalid))
    broker.cancel(m['id'])
    with pytest.raises(ValueError,match='generation'):broker.call_tool(call(m,'prepare_skill_job',{'node_id':'dock'}))
    with pytest.raises(ValueError,match='Stale'):broker.create_mission('old',[evidence(db,ttl=-.1).id])
    with pytest.raises(PermissionError):broker.create_mission('real',m['snapshot_ids'],mode='real')


def test_graph_conflicts_roles_and_verified_dependencies(db):
    a=mission(db);b=mission(db)
    for m in (a,b):broker.call_tool(call(m,'propose_graph',graph(m)))
    assert broker.call_tool(call(a,'prepare_skill_job',{'node_id':'dock'}))['result']['status']=='prepared'
    with pytest.raises(ValueError,match='owner conflict'):broker.call_tool(call(b,'prepare_skill_job',{'node_id':'dock'}))
    with pytest.raises(PermissionError,match='Role'):broker.call_tool(call(a,'prepare_skill_job',{'node_id':'dock'},role='SceneObserver'))
    invalid=graph(a);invalid['nodes'][0]['depends_on']=['dock']
    with pytest.raises(ValueError,match='cycle'):TaskGraph.model_validate(invalid)
    broker.cancel(a['id']);assert broker.call_tool(call(b,'prepare_skill_job',{'node_id':'dock'}))['result']['status']=='prepared'


def test_restart_unknown_does_not_replay_or_resume(db):
    m=mission(db);r=call(m,'list_skills')
    with catalog.connect() as c:
        c.execute('INSERT INTO cognitive_requests VALUES(?,?,?,?,?,?,?)',(r['request_id'],m['id'],m['decision_id'],store.digest(r),'running',None,time.time()))
    store.recover_interrupted()
    assert broker.call_tool(r)['state']=='unknown'
    assert store.mission(m['id'])['phase']=='unknown'


def test_memory_evidence_and_independent_verification(db):
    case=candidate(db);store.add_case(case)
    assert store.cases()[0]['status']=='candidate'
    for index in (1,2):
        payload={'episode':f'episode-{index}','compatibility_sha256':store.digest(case.compatibility),
                 'helper':'success','recipient':'success','task':'success'}
        e=evidence(db,source='fixed_outcome_verifier',payload=payload)
        for repeat in range(2):
            status=store.verify_case(case.id,payload['episode'],helper='success',recipient='success',task='success',evidence_ids=[e.id],verifier_version='fixed-v1')
            assert status==('candidate' if index==1 else 'supported_in_scope')
    skill=registry.get(case.compatibility.recipient_skill_id)
    found=store.retrieve('dock','target_visible',skill,'fixed_dock','sim');assert found['accepted'][0]['empirically_supported']
    incompatible=skill.model_copy(update={'camera_sha256':'f'*64})
    assert not store.retrieve('dock','target_visible',incompatible,'fixed_dock','sim')['accepted']
    assert store.retrieve('dock','target_visible',incompatible,'fixed_dock','sim',comparison='M2')['accepted']
    assert not store.retrieve('dock','target_visible',skill,'fixed_dock','sim',comparison='M1')['memory_enabled']
    payload={'episode':'negative-episode','compatibility_sha256':store.digest(case.compatibility),
             'helper':'success','recipient':'failure','task':'failure'}
    e=evidence(db,source='fixed_outcome_verifier',payload=payload)
    assert store.verify_case(case.id,payload['episode'],helper='success',recipient='failure',task='failure',evidence_ids=[e.id],verifier_version='fixed-v1')=='contradicted'
    assert not store.retrieve('dock','target_visible',skill,'fixed_dock','sim')['accepted']
    with catalog.connect() as c:assert c.execute('SELECT count(*) FROM cognitive_verifications').fetchone()[0]==3


def test_no_model_self_success_and_evidence_immutability(db):
    case=candidate(db)
    with pytest.raises(ValueError,match='independent'):store.add_case(case.model_copy(update={'task_outcome':'success'}))
    e=evidence(db)
    with pytest.raises(ValueError,match='immutable'):
        store.put_evidence(e.model_copy(update={'payload':{'new':1},'content_sha256':store.digest({'new':1})}))
    with pytest.raises(ValueError,match='hash mismatch'):store.put_evidence(e.model_copy(update={'content_sha256':'0'*64}))
    real=evidence(db,domain='real')
    with pytest.raises(ValueError,match='domain'):broker.create_mission('simulation',[real.id])


def test_unknown_conditions_owner_lock_and_late_answer(db,monkeypatch):
    e=evidence(db,payload={});m=broker.create_mission('unknown pose',[e.id])
    broker.call_tool(call(m,'propose_graph',graph(m)))
    with pytest.raises(ValueError,match='Preconditions'):broker.call_tool(call(m,'prepare_skill_job',{'node_id':'dock'}))
    fresh=broker.refresh_snapshot(m['id'],[evidence(db).id])
    with resources.ResourceLease('tank'):
        with pytest.raises(resources.ResourceBusy):broker.call_tool(call(fresh,'prepare_skill_job',{'node_id':'dock'}))
    original=broker._read_tool
    def late(request,mission):
        result=original(request,mission);broker.cancel(mission['id']);return result
    monkeypatch.setattr(broker,'_read_tool',late)
    with pytest.raises(ValueError,match='Late tool result'):broker.call_tool(call(fresh,'list_skills'))


def test_closed_args_and_four_step_budget(db):
    m=mission(db)
    with pytest.raises(ValueError):broker.call_tool(call(m,'list_skills',{'shell':'anything'}))
    for _ in range(3):broker.call_tool(call(m,'list_skills'))
    with pytest.raises(ValueError,match='budget'):broker.call_tool(call(m,'list_skills'))


def test_conflicting_conditions_and_expired_execution_claim(db):
    good=evidence(db)
    bad=evidence(db,payload={'conditions':{'recipient_stationary':False,'fixed_scene':True}})
    m=broker.create_mission('conflicting observations',[bad.id,good.id])
    broker.call_tool(call(m,'propose_graph',graph(m)))
    with pytest.raises(ValueError,match='Preconditions'):
        broker.call_tool(call(m,'prepare_skill_job',{'node_id':'dock'}))
    fresh=broker.refresh_snapshot(m['id'],[good.id])
    with pytest.raises(ValueError,match='reservation missing'):
        with broker.execution_resources(m['id']):pass
    broker.call_tool(call(fresh,'prepare_skill_job',{'node_id':'dock'}))
    with broker.execution_resources(m['id']) as admitted:assert admitted['id']==m['id']
    broker.cancel(m['id'])
    with pytest.raises(PermissionError):
        with broker.execution_resources(m['id']):pass


def test_frozen_memory_does_not_gain_later_cases_or_promotions(db):
    first=candidate(db);store.add_case(first);snapshot=store.memory_snapshot()
    second=candidate(db);store.add_case(second)
    skill=registry.get(first.compatibility.recipient_skill_id)
    result=store.retrieve('dock','target_visible',skill,'fixed_dock','sim',snapshot=snapshot)
    assert [x['case']['id'] for x in result['accepted']]==[first.id]
    assert result['accepted'][0]['case']['status']=='candidate'


def test_restarted_planned_mission_needs_new_admission(db):
    m=mission(db);broker.call_tool(call(m,'propose_graph',graph(m)))
    broker.call_tool(call(m,'prepare_skill_job',{'node_id':'dock'}))
    store.recover_interrupted()
    assert store.mission(m['id'])['phase']=='unknown'
    with pytest.raises(PermissionError):
        with broker.execution_resources(m['id']):pass
