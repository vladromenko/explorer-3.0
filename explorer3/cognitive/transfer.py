"""S3: transfer a memory hypothesis, never another robot's motor actions.

Compatibility belongs to the new recipient. A successful historical donor case
cannot by itself validate either a new scene or a different policy.
"""
import time
import uuid
from . import registry,store
from .schemas import AssistanceCase,CompatibilityRecord


def prepare(case_id,target_skill_id,evidence_ids):
    case=next((c for c in store.cases() if c['id']==case_id),None)
    if case is None:raise ValueError('Unknown source memory')
    target=registry.get(target_skill_id)
    if target.robot==case['recipient']:raise ValueError('Для переноса выберите другую платформу')
    if case['status'] in ('contradicted','stale'):raise ValueError('Источник опровергнут или устарел')
    if case['compatibility']['goal_id'] not in target.goals:raise ValueError('Новый навык не принимает цель исходного случая')
    domain=case['domain']
    if (target.real_admission if domain=='real' else target.sim_admission)!='available':raise ValueError('Навык получателя недоступен в этом режиме')
    items=store.evidence(evidence_ids,domain=domain,fresh=False)
    if not items or not any(e.robot==target.robot for e in items):raise ValueError('Нужно собственное наблюдение нового получателя')
    for item in items:store.validate_evidence_artifact(item)
    now=time.time();fresh=all(not e.payload.get('archived') and e.created_at<=now<e.expires_at for e in items)
    compatible=store.compatibility(case,target,case['compatibility']['goal_id'],domain)
    # Different robot/policy/camera is expected, but must be revalidated, never
    # erased from the report or copied into a supported scope.
    return {'schema':'memory_transfer_v1','source_case_id':case_id,'source_case_sha256':store.digest(case),
        'donor':case['recipient'],'donor_episode':case['recipient_episode'],'source_helper_episode':case['donor_episode'],
        'recipient':target.robot,'recipient_skill':target.model_dump(),'domain':domain,
        'task':case['task'],'effect':case['effect'],'assistance_type':case['assistance_type'],
        'expected_effect':case['expected_effect'],'source_scope_mismatches':compatible,
        'evidence_ids':evidence_ids,'evidence_hashes':{e.id:e.content_sha256 for e in items},
        'fresh_recipient_observation':fresh,'status':'hypothesis_requires_new_continuation_check',
        'recipient_weights_updated':False,'execution_started':False,'source_case_used_as_permission':False}


def validate(value):
    source=next((c for c in store.cases() if c['id']==value['source_case_id']),None)
    if source is None or store.digest(source)!=value['source_case_sha256']:raise ValueError('Source memory changed; prepare a new transfer')
    target=registry.get(value['recipient_skill']['skill_id'])
    if target.model_dump()!=value['recipient_skill']:raise ValueError('Recipient skill changed')
    for item in store.evidence(value['evidence_ids'],domain=value['domain'],fresh=False):
        store.validate_evidence_artifact(item)
        if value['evidence_hashes'].get(item.id)!=item.content_sha256:raise ValueError('Transfer evidence changed')
    return target


def record_continuation(value,episode_id,*,mission_id):
    """Only a finalized matching mission step supplies the recipient episode."""
    from explorer3 import catalog
    target=validate(value);mission=store.mission(mission_id)
    if mission['domain']!='real' or mission['phase'] not in ('completed','cancelled') or mission['plan'].get('memory_transfer')!=value:
        raise ValueError('No finalized mission linked to this memory transfer')
    outcomes=[r for r in mission['node_outcomes'] if r['episode_id']==episode_id]
    row=next((r for r in catalog.rows('episodes') if r['id']==episode_id),None)
    if len(outcomes)!=1 or row is None or row['profile']!=target.profile or row['outcome']!=outcomes[0]['outcome']:
        raise ValueError('Recipient episode does not confirm the recorded continuation')
    if not any(n['skill_id']==target.skill_id and n['id']==outcomes[0]['node'] for n in mission['plan']['graph']['nodes']):
        raise ValueError('Episode was not produced by the selected recipient skill')
    ident=store.digest({'transfer':value,'episode':episode_id,'mission':mission_id})
    if any(c['id']==ident for c in store.cases()):return ident
    scope=CompatibilityRecord(recipient_skill_id=target.skill_id,recipient_version=target.version,
        policy_sha256=target.policy_sha256,camera_sha256=target.camera_sha256,
        calibration_sha256=target.calibration_sha256,domain=value['domain'],goal_id=target.goals[0],
        checked_conditions=['finalized original episode from selected trial'],
        unknown_conditions=['causal benefit of memory not measured','helper effect in new scene not independently verified'],
        evidence_ids=value['evidence_ids'])
    item=AssistanceCase(id=ident,task=value['task'],effect=value['effect'],domain=value['domain'],
        donor=value['donor'],recipient=target.robot,donor_episode=value['donor_episode'],recipient_episode=episode_id,
        reason='Cross-platform memory hypothesis; independent recipient continuation recorded',
        before_evidence_ids=value['evidence_ids'],compatibility=scope,assistance_type=value['assistance_type'],
        expected_effect=value['expected_effect'],actions_provenance='policy' if target.kind=='learned' else 'scripted',
        actual_effect={'source_case_id':value['source_case_id'],'source_case_sha256':value['source_case_sha256'],
            'mission_id':mission_id,'observed_recipient_outcome':row['outcome'],'source':'operator_label_and_saved_episode',
            'weights_updated':False,'causal_transfer_effect':'unknown'})
    store.add_case(item)
    from .schemas import ObservationEvidence
    payload={'helper':'unknown','recipient':row['outcome'],'task':mission['task_outcome'],
        'episode':episode_id,'compatibility_sha256':store.digest(scope),
        'measurement_source':'operator_label_and_finalized_original_episode','mission_id':mission_id}
    now=time.time()
    observation=store.put_evidence(ObservationEvidence(id=ident+'v',domain='real',robot=target.robot,
        source='fixed_outcome_verifier',sample_id=episode_id,frame_id=target.robot+'/recorded_outcome',
        received_timestamp_ns=time.time_ns(),clock_domain='homebrain_wall',created_at=now,expires_at=now+1,
        status='observed',content_sha256=store.digest(payload),payload=payload))
    store.verify_case(ident,episode_id,helper='unknown',recipient=row['outcome'],task=mission['task_outcome'],
        evidence_ids=[observation.id],verifier_version='original_episode_operator_outcome_v1')
    store.event(mission_id,'memory_transfer_continuation',{'case_id':ident,'donor_episode':value['donor_episode'],
        'source_case_id':value['source_case_id'],'recipient_episode':episode_id,'outcome':row['outcome'],
        'helper_outcome':'unknown','scientific_effect_established':False})
    return ident
