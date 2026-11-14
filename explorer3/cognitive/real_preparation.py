"""AG3-AG5 shadow preparation over existing profiles/trials, never motor commands."""
import time
from explorer3 import trials
from explorer3.paths import DATA
from explorer3.training_drafts import save_content
from . import registry,store


def prepare(profile,evidence_ids,trial_id=None):
    card=registry.get(profile+'.manual.v1')
    items=store.evidence(evidence_ids,domain='real',fresh=False)
    if not items or any(item.robot!=card.robot for item in items):raise ValueError('Evidence belongs to another robot')
    for item in items:store.validate_evidence_artifact(item)
    now=time.time();archived=any(item.payload.get('archived') for item in items)
    recent=not archived and all(item.created_at<=now<item.expires_at for item in items)
    problems=[]
    if not recent:problems.append('Требуется свежее наблюдение; архив не разрешает движение')
    policy=None
    if trial_id:
        policy=trials.load(trial_id)
        if policy['profile']!=profile:raise ValueError('Trial profile mismatch')
        if policy['kind']=='policy':trials.verify_policy_artifacts(policy)
        try:trials.require_physical(policy)
        except (PermissionError,ValueError) as exc:problems.append(str(exc))
    else:problems.append('Автономный checkpoint не выбран; доступен сбор показов оператором')
    if card.robot=='tank' and (DATA/'validation/tank-motion-incident.json').exists():
        problems.append('Tank: инцидент USB/STOP ещё не закрыт')
    result={'schema':'real_shadow_preparation_v1','robot':card.robot,'profile':profile,
        'skill':card.model_dump(),'mode':'shadow','evidence_ids':evidence_ids,
        'evidence_hashes':{e.id:e.content_sha256 for e in items},'artifact_hashes':{e.id:e.artifact_sha256 for e in items},
        'recorded_observations':archived,'live_observations':recent,'trial_id':trial_id,
        'trial_sha256':trials.plan_hash(policy) if policy else None,'problems':problems,
        'control_url':'/rosmaster/control' if card.robot=='rosmaster' else '/',
        'training_url':'/training/prepare','created_at':now,'motion_authorized':False,'execution_started':False,
        'task_outcome':'unknown','next_step':'collect_demonstrations' if policy is None else 'review_existing_trial',
        'conditions':['Fresh sensors and owner are rechecked by the existing physical executor',
            'This shadow plan grants no permission and never dispatches a motor command']}
    ident=save_content(DATA/'cognitive/real-preparations',result)
    store.event(None,'real_shadow_prepared',{'id':ident,'robot':card.robot,'evidence_ids':evidence_ids})
    return ident,result
