"""Deterministic admission and deduplication. No physical or trainer dispatcher."""
import contextlib
import json
import time
import uuid
from explorer3 import catalog
from explorer3.resources import ResourceLease
from . import store,registry
from .schemas import ToolCall,TaskGraph,AssistanceCase
from .tool_args import SCHEMAS

PERMISSIONS={
 'MissionCoordinator':{'get_snapshot','list_skills','retrieve_cases','propose_graph','prepare_skill_job','request_human'},
 'SceneObserver':{'get_snapshot','request_human'},
 'AssistanceBroker':{'get_snapshot','list_skills','retrieve_cases','propose_help','request_human'},
 'MemoryCurator':{'retrieve_cases','add_memory_candidate'},
 'OutcomeVerifier':{'get_snapshot','verify_postcondition','request_human'},
 'LearningAndExperimentCurator':{'prepare_learning_round','retrieve_cases','request_human'}}


def create_mission(task,snapshot_ids,*,mode='sim'):
    if mode!='sim':raise PermissionError('AG0–AG2: real mode is blocked')
    if not isinstance(task,str) or not 1<=len(task)<=600:raise ValueError('Коротко опишите задачу')
    store.evidence(snapshot_ids,domain=mode)
    value={'id':uuid.uuid4().hex,'task':task,'domain':mode,'generation':0,'decision_id':uuid.uuid4().hex,
        'snapshot_ids':snapshot_ids,'model_steps':0,'phase':'planning','created_at':time.time(),'graph':None,
        'training_authorized':False,'real_authorized':False,'memory_snapshot':store.memory_snapshot()}
    with catalog.connect() as db:
        db.execute('INSERT INTO cognitive_missions VALUES(?,?)',(value['id'],store.encoded(value)))
        store.event(value['id'],'mission_created',value,db)
    return value


def cancel(mission_id,reason='operator_cancelled'):
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT body FROM cognitive_missions WHERE id=?',(mission_id,)).fetchone()
        if row is None:raise ValueError('Unknown mission')
        value=json.loads(row['body'])
        if value['phase'] not in ('cancelled','completed'):
            value.update(phase='cancelled',generation=value['generation']+1,error=reason)
            db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(store.encoded(value),mission_id))
            db.execute('DELETE FROM cognitive_claims WHERE mission_id=?',(mission_id,))
            store.event(mission_id,'cancelled',{'reason':reason,'generation':value['generation']},db)
    return value


def refresh_snapshot(mission_id,snapshot_ids):
    old=store.mission(mission_id);store.evidence(snapshot_ids,domain=old['domain'])
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current=json.loads(db.execute('SELECT body FROM cognitive_missions WHERE id=?',(mission_id,)).fetchone()['body'])
        if current['phase'] in ('cancelled','unknown','completed'):raise ValueError('Mission no longer accepts observations')
        current.update(snapshot_ids=snapshot_ids,generation=current['generation']+1,decision_id=uuid.uuid4().hex,model_steps=0)
        db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(store.encoded(current),mission_id))
        db.execute('DELETE FROM cognitive_claims WHERE mission_id=?',(mission_id,))
        store.event(mission_id,'snapshot_changed',{'ids':snapshot_ids,'generation':current['generation']},db)
    return current


def _guard(call,mission,db):
    if call.mode!='sim' or mission['domain']!='sim':raise PermissionError('Real actuation blocked during AG0–AG2')
    if call.generation!=mission['generation'] or call.decision_id!=mission['decision_id']:raise ValueError('Stale decision generation')
    if mission['phase'] in ('cancelled','unknown','completed'):raise ValueError('Mission is '+mission['phase'])
    if call.expires_at<=time.time() or call.expires_at>time.time()+120:raise ValueError('Expired or excessive request deadline')
    if call.tool not in PERMISSIONS[call.role]:raise PermissionError('Role cannot call this tool')
    if set(call.snapshot_ids)!=set(mission['snapshot_ids']):raise ValueError('World snapshot changed')
    store.evidence(call.snapshot_ids,domain=call.mode)
    count=db.execute('SELECT count(*) FROM cognitive_requests WHERE mission_id=? AND decision_id=?',
                     (call.mission_id,call.decision_id)).fetchone()[0]
    if count+mission.get('model_steps',0)>=4:raise ValueError('Decision budget exhausted; deterministic fallback required')


def call_tool(value):
    call=ToolCall.model_validate(value);fingerprint=store.digest(call)
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('SELECT * FROM cognitive_requests WHERE id=?',(call.request_id,)).fetchone()
        if old:
            if old['fingerprint']!=fingerprint:raise ValueError('request_id reused with different inputs')
            return {'request_id':call.request_id,'state':old['state'],'replayed':True,
                    'result':json.loads(old['result']) if old['result'] else None}
        row=db.execute('SELECT body FROM cognitive_missions WHERE id=?',(call.mission_id,)).fetchone()
        if row is None:raise ValueError('Unknown mission')
        mission=json.loads(row['body']);_guard(call,mission,db)
        db.execute('INSERT INTO cognitive_requests VALUES(?,?,?,?,?,?,?)',
            (call.request_id,call.mission_id,call.decision_id,fingerprint,'running',None,time.time()))
        store.event(call.mission_id,'tool_proposed',call,db)
    started=time.monotonic()
    try:
        result=_read_tool(call,mission)
        with catalog.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=json.loads(db.execute('SELECT body FROM cognitive_missions WHERE id=?',(call.mission_id,)).fetchone()['body'])
            if current['generation']!=call.generation or current['phase'] in ('cancelled','unknown') or time.time()>=call.expires_at:
                raise ValueError('Late tool result discarded after cancellation or world change')
            if call.tool=='propose_graph':
                current.update(graph=result['graph'],phase='planned')
                db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(store.encoded(current),call.mission_id))
            if call.tool=='prepare_skill_job':
                _reserve(db,call,registry.get(result['skill_id']).resources)
            if call.tool=='add_memory_candidate':
                ident=store.add_case(result['candidate'],db=db)
                result={'case_id':ident,'status':'candidate','verification_required':True}
            db.execute("UPDATE cognitive_requests SET state='completed',result=? WHERE id=?",(store.encoded(result),call.request_id))
            store.event(call.mission_id,'tool_completed',{'request_id':call.request_id,'result':result,'latency_s':time.monotonic()-started},db)
        return {'request_id':call.request_id,'state':'completed','replayed':False,'result':result}
    except Exception as exc:
        with catalog.connect() as db:
            db.execute("UPDATE cognitive_requests SET state='rejected',result=? WHERE id=?",(store.encoded({'error':str(exc)}),call.request_id))
            store.event(call.mission_id,'tool_rejected',{'request_id':call.request_id,'error':str(exc)},db)
        raise


def _read_tool(call,mission):
    args=SCHEMAS[call.tool].model_validate(call.arguments).model_dump(mode='json')
    if call.tool=='get_snapshot':return {'evidence':[x.model_dump() for x in store.evidence(call.snapshot_ids,domain=call.mode)]}
    if call.tool=='list_skills':return {'skills':[c.model_dump() for c in registry.cards().values()]}
    if call.tool=='retrieve_cases':
        skill=registry.get(args['recipient_skill_id'])
        observed=store.evidence(call.snapshot_ids,domain=call.mode)
        context={key:observed[0].payload[key] for key in ('scene','model_sha256') if observed and key in observed[0].payload}
        return store.retrieve(args['task'],args['effect'],skill,args['goal_id'],call.mode,comparison=args.get('comparison','M3'),snapshot=mission['memory_snapshot'],context=context)
    if call.tool=='propose_graph':
        graph=TaskGraph.model_validate(args);registry.validate_graph(graph,call.mode)
        if not set(graph.evidence_ids)<=set(call.snapshot_ids):raise ValueError('Plan cites evidence outside its snapshot')
        return {'graph':graph.model_dump(),'accepted':True,'execution_started':False}
    if call.tool=='prepare_skill_job':
        if not mission.get('graph'):raise ValueError('A validated graph is required first')
        graph=TaskGraph.model_validate(mission['graph']);registry.validate_graph(graph,call.mode)
        node=next((n for n in graph.nodes if n.id==args.get('node_id')),None)
        if node is None:raise ValueError('Unknown graph step')
        completed=mission.get('verified_nodes',[])
        if not set(node.depends_on)<=set(completed):raise ValueError('Dependencies need measured verification')
        card=registry.get(node.skill_id)
        observations=store.evidence(call.snapshot_ids,domain=call.mode)
        measured={}
        for item in observations:
            if item.robot==card.robot and item.status=='observed':
                for key,value in item.payload.get('conditions',{}).items():
                    measured.setdefault(key,[]).append(value)
        conditions={key:all(value is True for value in values) for key,values in measured.items()}
        missing=[name for name in card.preconditions if name!='fresh_evidence' and conditions.get(name) is not True]
        if missing:raise ValueError('Preconditions not observed: '+', '.join(missing))
        return {'job_id':store.digest({'mission':call.mission_id,'node':node.id,'generation':call.generation}),
            'skill_id':node.skill_id,'node_id':node.id,'domain':'sim','status':'prepared','execution_started':False}
    if call.tool=='propose_help':
        options=args.get('candidates',[])
        if not 1<=len(options)<=3 or any(x not in ('continue','observe','recovery','human') for x in options):raise ValueError('At most three registered assistance types')
        return {'candidates':options,'selection':'observe' if 'observe' in options else options[0],
                'baseline':'fixed_rules_v1','gain':None,'gain_status':'unavailable','execution_started':False}
    if call.tool=='verify_postcondition':
        measurements=store.evidence(args.get('evidence_ids',[]),domain=call.mode)
        verified=[e for e in measurements if e.source=='fixed_outcome_verifier']
        return {'outcome':'unknown' if not verified else verified[-1].payload.get('task','unknown'),
                'evidence_ids':[e.id for e in verified],'model_is_ground_truth':False}
    if call.tool=='add_memory_candidate':
        case=AssistanceCase.model_validate(args)
        if not set(case.before_evidence_ids)<=set(call.snapshot_ids):raise ValueError('Memory cites another snapshot')
        # The accepted proposal is materialized by the deterministic curator after
        # this call, so cancellation cannot race a mutation from a model tool.
        if case.status!='candidate' or any(x!='unknown' for x in (case.helper_outcome,case.recipient_outcome,case.task_outcome)):
            raise ValueError('Model cannot verify its own memory')
        return {'candidate':case.model_dump(),'status':'proposed','verification_required':True}
    if call.tool=='prepare_learning_round':
        from explorer3.training_drafts import prepare
        if set(args)-{'profile','policy','episode_ids','steps','batch','seed','device'}:raise ValueError('Unknown training preparation field')
        ident,draft=prepare(args['profile'],args['policy'],args['episode_ids'],steps=args['steps'],batch=args['batch'],seed=args['seed'],device=args['device'])
        return {'status':'prepared','draft_id':ident,'draft':draft,'training_authorized':False,'training_started':False}
    if call.tool=='request_human':return {'question':str(args.get('question',''))[:600],'status':'pending','permission_granted':False}
    raise ValueError('Tool not implemented')


def _reserve(db,call,resources):
    now=time.time()
    with contextlib.ExitStack() as stack:
        for name in sorted(set(resources)):
            stack.enter_context(ResourceLease(name))
    db.execute('DELETE FROM cognitive_claims WHERE expires<=?',(now,))
    db.execute("DELETE FROM cognitive_claims WHERE mission_id IN (SELECT id FROM cognitive_missions WHERE json_extract(body,'$.phase') IN ('completed','cancelled','unknown'))")
    for resource in sorted(set(resources)):
        owner=db.execute('SELECT * FROM cognitive_claims WHERE resource=?',(resource,)).fetchone()
        if owner and (owner['mission_id']!=call.mission_id or owner['generation']!=call.generation):raise ValueError('Resource owner conflict: '+resource)
        db.execute('INSERT OR REPLACE INTO cognitive_claims VALUES(?,?,?,?)',(resource,call.mission_id,call.generation,call.expires_at))


@contextlib.contextmanager
def execution_resources(mission_id):
    mission=store.mission(mission_id)
    if mission['domain']!='sim' or mission['phase']!='planned':raise PermissionError('Mission not admitted for simulation')
    cards=registry.validate_graph(TaskGraph.model_validate(mission['graph']),'sim')
    with contextlib.ExitStack() as stack:
        # Reuse the same process locks as the existing executor, in fixed order.
        for name in sorted({'compute','cognitive-mission',*[r for c in cards for r in c.resources]}):
            stack.enter_context(ResourceLease(name))
        current=store.mission(mission_id)
        if current['generation']!=mission['generation'] or current['phase']!='planned':
            raise ValueError('Mission changed while acquiring execution resources')
        store.evidence(current['snapshot_ids'],domain='sim')
        with catalog.connect() as db:
            for resource in {r for card in cards for r in card.resources}:
                owner=db.execute('SELECT * FROM cognitive_claims WHERE resource=?',(resource,)).fetchone()
                if owner is None or owner['mission_id']!=mission_id or owner['generation']!=current['generation'] or owner['expires']<=time.time():
                    raise ValueError('Execution resource reservation missing or expired: '+resource)
        yield current


@contextlib.contextmanager
def execution_step(mission_id,node_id):
    """Fixed job dispatcher only; the existing Jobs parent holds compute already."""
    mission=store.mission(mission_id)
    if mission['domain']!='sim' or mission['phase']!='planned':raise PermissionError('Mission not admitted')
    graph=TaskGraph.model_validate(mission['graph']);registry.validate_graph(graph,'sim')
    node=next((n for n in graph.nodes if n.id==node_id),None)
    if node is None or not set(node.depends_on)<=set(mission.get('verified_nodes',[])):raise ValueError('Unverified dependency')
    card=registry.get(node.skill_id)
    with contextlib.ExitStack() as stack:
        for name in sorted(card.resources):stack.enter_context(ResourceLease(name))
        current=store.mission(mission_id)
        if current['generation']!=mission['generation'] or current['phase']!='planned':raise ValueError('Execution was cancelled')
        store.evidence(current['snapshot_ids'],domain='sim')
        with catalog.connect() as db:
            for resource in card.resources:
                claim=db.execute('SELECT * FROM cognitive_claims WHERE resource=?',(resource,)).fetchone()
                if claim is None or claim['mission_id']!=mission_id or claim['generation']!=current['generation'] or claim['expires']<=time.time():
                    raise ValueError('Execution reservation missing or expired')
        yield current



def record_model_steps(mission_id,generation,steps):
    """Trusted worker report; model steps and broker calls share one budget."""
    if type(steps) is not int or not 1<=steps<=2:raise ValueError('Invalid model step report')
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current=json.loads(db.execute('SELECT body FROM cognitive_missions WHERE id=?',(mission_id,)).fetchone()['body'])
        if current['phase'] in ('cancelled','unknown','completed') or current['generation']!=generation:
            raise ValueError('Late model usage report discarded')
        count=db.execute('SELECT count(*) FROM cognitive_requests WHERE mission_id=? AND decision_id=?',
                         (mission_id,current['decision_id'])).fetchone()[0]
        if current.get('model_steps',0) or count+steps>4:raise ValueError('Model decision budget exhausted or already recorded')
        current['model_steps']=steps
        db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(store.encoded(current),mission_id))
        store.event(mission_id,'model_usage',{'decision_id':current['decision_id'],'steps':steps},db)
