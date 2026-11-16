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
        from .assistance import select
        return select(args['candidates'],store.evidence(call.snapshot_ids,domain=call.mode))
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


ACQUISITION_TITLES = {
    'needs_demonstration': 'нужен пример захвата SO-101 с измеренной геометрией',
    'transfer_preparation': 'опыт SO-101 найден; перенос проверяется',
    'data_preparation': 'готовятся собственные показы ROSMASTER',
    'awaiting_training_budget': 'данные готовы; нужен согласованный бюджет обучения',
    'training': 'ROSMASTER обучается',
    'validation': 'проверяется собственная политика ROSMASTER',
    'awaiting_physical_trial': 'ждёт разрешения испытания',
    'accepted': 'новый навык принят',
    'rejected': 'перенос недоступен в этих условиях',
    'cancelled': 'запрос навыка отменён',
}


def prepare_acquisition(mission_id, task, robot, object_class, assessed, *, domain, context=None):
    """Persist the existing fleet mission's nonmoving wait, deduplicated across requests.

    Preparation only: no new worker, trainer, hardware lease or permission. Existing
    Jobs and prepared training drafts execute later stages after a concrete budget.
    """
    from explorer3 import knowledge
    context = context or {}
    if domain not in ('real', 'sim') or robot != 'rosmaster':
        raise ValueError('Unsupported acquisition recipient')
    if assessed.get('registered_skill', {}).get('state') == 'available' and not assessed.get('readiness', {}).get('physically_infeasible'):
        return {'state': 'accepted', 'skill_id': assessed['registered_skill']['skill_id'], 'training_started': False}
    source_domain = context.get('source_domain', 'real')
    if source_domain not in ('real', 'sim') or (domain == 'real' and source_domain != 'real'):
        raise ValueError('A simulated donor cannot become a physically verified source')
    candidates = []
    for item in knowledge._load():
        sha = item.get('skill_packet_sha256')
        if sha and item.get('object_class') == object_class and item.get('domain') == source_domain:
            packet = knowledge.load_packet(sha)
            from explorer3.skill_packets import admission
            if admission(packet, domain=source_domain)['eligible']:
                candidates.append(sha)
    infeasible = assessed.get('readiness', {}).get('physically_infeasible', False)
    state = 'rejected' if infeasible else 'transfer_preparation' if candidates else 'needs_demonstration'
    fingerprint = store.digest({'kind': 'skill_acquisition', 'recipient': robot, 'object_class': object_class,
                                'domain': domain, 'context': context, 'source_domain': source_domain})
    store.initialize()
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute("SELECT * FROM cognitive_requests WHERE fingerprint=? AND state != 'cancelled' ORDER BY created LIMIT 1", (fingerprint,)).fetchone()
        if old:
            value = json.loads(old['result'])
            if value['state'] == 'needs_demonstration' and candidates:
                value.update(state='transfer_preparation', title=ACQUISITION_TITLES['transfer_preparation'], candidate_packets=candidates,
                             operator_next='Опыт найден; сначала проверяется перенос, обучение ещё не запускается.')
                db.execute('UPDATE cognitive_requests SET state=?,result=? WHERE id=?', (value['state'], store.encoded(value), old['id']))
            store.event(mission_id, 'acquisition_reused', {'request_id': old['id']}, db)
        else:
            value = {'kind': 'skill_acquisition', 'id': uuid.uuid4().hex, 'mission_id': mission_id,
                'task': task, 'recipient': robot, 'object_class': object_class, 'domain': domain,
                'source_domain': source_domain, 'context': context, 'state': state, 'title': ACQUISITION_TITLES[state],
                'candidate_packets': candidates, 'selected_packet': None, 'jobs': [], 'assessed': assessed,
                'training_started': False, 'motion_authorized': False, 'resume_requires_fresh_scene': True,
                'operator_next': ('Три пробных показа SO-101: подход, захват, подъём и удержание; '
                    'верхняя камера и геометрия TCP должны быть проверены. Мобильные роботы не нужны.') if not candidates else
                    'Сначала проверить перенос одного выбранного пакета; обучение и движение ещё не начаты.',
                'created_at': time.time()}
            if infeasible:
                value['operator_next'] = assessed['reasons'].get('physically_infeasible', 'Изменить сцену или исполнителя')
            db.execute('INSERT INTO cognitive_requests VALUES(?,?,?,?,?,?,?)',
                       (value['id'], mission_id, 'acquisition', fingerprint, state, store.encoded(value), time.time()))
            store.event(mission_id, 'acquisition_prepared', value, db)
        row = db.execute('SELECT body FROM cognitive_missions WHERE id=?', (mission_id,)).fetchone()
        mission = json.loads(row['body']) if row else {'id': mission_id, 'task': task, 'domain': domain, 'generation': 0}
        mission.update(phase='awaiting_skill', acquisition_request=value['id'], training_authorized=False,
                       real_authorized=False, auto_resume_motion=False, resume_requires_fresh_scene=True)
        db.execute('INSERT INTO cognitive_missions VALUES(?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                   (mission_id, store.encoded(mission)))
    return value


def acquisition_requests():
    with catalog.connect() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cognitive_requests'").fetchone()
        rows = db.execute("SELECT result FROM cognitive_requests WHERE decision_id='acquisition' ORDER BY created DESC LIMIT 20").fetchall() if exists else []
    return [reconcile_acquisition(json.loads(row['result'])['id']) for row in rows if row['result']]


def cancel_acquisition(identifier):
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT * FROM cognitive_requests WHERE id=? AND decision_id='acquisition'", (identifier,)).fetchone()
        if row is None:
            raise ValueError('Unknown acquisition request')
        value = json.loads(row['result'])
        if value.get('jobs'):
            raise ValueError('Cancel linked training through its existing queue first')
        value.update(state='cancelled', title=ACQUISITION_TITLES['cancelled'])
        db.execute('UPDATE cognitive_requests SET state=?,result=? WHERE id=?', ('cancelled', store.encoded(value), identifier))
        for linked in db.execute("SELECT id,body FROM cognitive_missions WHERE json_extract(body,'$.acquisition_request')=?", (identifier,)).fetchall():
            mission = json.loads(linked['body'])
            mission.update(phase='cancelled', generation=mission.get('generation', 0) + 1, auto_resume_motion=False)
            db.execute('UPDATE cognitive_missions SET body=? WHERE id=?', (store.encoded(mission), linked['id']))
            db.execute('DELETE FROM cognitive_claims WHERE mission_id=?', (linked['id'],))
        store.event(row['mission_id'], 'acquisition_cancelled', {'request_id': identifier}, db)
    return value


def acquisition(identifier):
    with catalog.connect() as db:
        row = db.execute("SELECT result FROM cognitive_requests WHERE id=? AND decision_id='acquisition'", (identifier,)).fetchone()
    if row is None:
        raise ValueError('Запрос навыка не найден')
    return json.loads(row['result'])


def _save_acquisition(db, value, event):
    value['title'] = ACQUISITION_TITLES[value['state']]
    db.execute('UPDATE cognitive_requests SET state=?,result=? WHERE id=?',
               (value['state'], store.encoded(value), value['id']))
    store.event(value['mission_id'], event, {'request_id': value['id'], 'state': value['state']}, db)


def select_acquisition_packet(identifier, packet_sha):
    """Choose immutable evidence, without a collection job or hardware command."""
    from explorer3 import knowledge
    from explorer3.skill_packets import admission
    packet = knowledge.load_packet(packet_sha)
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT result FROM cognitive_requests WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Запрос навыка не найден')
        value = json.loads(row['result'])
        if value['state'] not in ('transfer_preparation', 'data_preparation') or value.get('jobs'):
            raise ValueError('Выбор пакета сейчас закрыт')
        if packet_sha not in value['candidate_packets'] or not admission(packet, domain=value['source_domain'])['eligible']:
            raise ValueError('Пакет не подходит этому запросу')
        if packet['object']['class'] != value['object_class'] or packet['object']['surface'] != 'rigid_box':
            raise ValueError('Предмет или механика пакета не поддерживаются')
        size = value['context'].get('size_m')
        if size is not None and (len(size) != 3 or any(abs(a / b - 1.) > .2 for a, b in zip(size, packet['object']['size_m']))):
            raise ValueError('Размеры выходят за область первого переноса')
        chosen = sorted(set(value.get('selected_packets', [])) | {packet_sha})
        value.update(selected_packet=chosen[0], selected_packets=chosen, state='data_preparation',
                     operator_next='Пакет выбран. Нужны собственные успешные показы ROSMASTER и независимые группы данных.')
        _save_acquisition(db, value, 'donor_packet_selected')
    return value


def record_transfer_attempt(identifier, report_path):
    """Link an actual imported simulator rollout, including failed attempts."""
    import hashlib
    from pathlib import Path
    from explorer3.paths import DATA
    from explorer3 import datasets
    path = Path(report_path).resolve()
    if not path.is_relative_to(DATA / 'simlab/skill-transfer') or not path.is_file():
        raise ValueError('Отчёт должен находиться в хранилище переноса Explorer')
    report = json.loads(path.read_text())
    imported = report['recipient_import']
    matches = [e for e in catalog.rows('episodes') if e['id'] == imported['episode_id']]
    if len(matches) != 1 or matches[0]['profile'] != 'rosmaster_floor_pick_sim':
        raise ValueError('Нет собственного эпизода ROSMASTER')
    episode = matches[0]
    if datasets.fingerprint(episode['dataset']) != imported['dataset_sha256']:
        raise ValueError('Сохранённый эпизод изменился')
    provenance = json.loads((Path(episode['dataset']) / 'provenance.json').read_text())
    lineage = provenance.get('hardware', {}).get('transfer_lineage', {})
    if lineage.get('packet_sha256') != report.get('packet_sha256') or report.get('physical_commands_sent') != 0:
        raise ValueError('Происхождение отчёта не совпадает с записью')
    accepted = (report['outcome'] == episode['outcome'] == 'success'
                and report.get('independent_verification', {}).get('accepted') is True)
    attempt = {'path': str(path.relative_to(DATA)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'packet_sha256': report['packet_sha256'], 'episode_id': episode['id'],
               'dataset_sha256': imported['dataset_sha256'], 'accepted': accepted,
               'reason': report.get('termination_reason'), 'seed': report['recipient_seed']}
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT result FROM cognitive_requests WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Запрос навыка не найден')
        value = json.loads(row['result'])
        if value['state'] != 'data_preparation' or report['packet_sha256'] not in value.get('selected_packets', []):
            raise ValueError('Результат не относится к выбранной подготовке')
        attempts = {a['episode_id']: a for a in value.get('transfer_attempts', [])}
        if episode['id'] in attempts and attempts[episode['id']] != attempt:
            raise ValueError('Существующий результат нельзя перезаписать')
        attempts[episode['id']] = attempt
        value['transfer_attempts'] = list(attempts.values())
        _save_acquisition(db, value, 'transfer_attempt_recorded')
    return value


def acquisition_training_lineage(identifier, plan):
    """A prepared plan may use only linked, verified recipient derivatives."""
    value = acquisition(identifier)
    if value['state'] not in ('data_preparation', 'awaiting_training_budget') or value.get('jobs'):
        raise ValueError('Подготовка нового обучения сейчас закрыта')
    if plan['profile'] != 'rosmaster_floor_pick_sim' or plan['domain'] != 'sim':
        raise ValueError('Первый план переноса использует собственную среду ROSMASTER')
    allowed = {a['episode_id']: a for a in value.get('transfer_attempts', []) if a['accepted']}
    entries = [e for group in plan['splits'].values() for e in group]
    if not entries or any(e['id'] not in allowed or e['sha256'] != allowed[e['id']]['dataset_sha256'] for e in entries):
        raise ValueError('В плане есть записи вне проверенного переноса')
    packets = sorted({allowed[e['id']]['packet_sha256'] for e in entries})
    from explorer3 import knowledge
    for sha in packets:
        if 'training' not in knowledge.load_packet(sha)['donor']['usage']['allowed']:
            raise ValueError('Использование источника для обучения не разрешено')
    return {'acquisition_request': identifier, 'donor_packet_shas': packets,
            'recipient_derivatives': sorted(e['id'] for e in entries), 'target_real_demonstrations': 0,
            'experiment_scope': 'simulation transfer mechanism; no physical low-data result implied'}


def record_acquisition_draft(identifier, draft_id, draft):
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT result FROM cognitive_requests WHERE id=?', (identifier,)).fetchone()
        value = json.loads(row['result'])
        if value['state'] not in ('data_preparation', 'awaiting_training_budget') or value.get('jobs'):
            raise ValueError('Запрос изменился во время подготовки')
        value.update(state='awaiting_training_budget', draft_id=draft_id,
                     budget={k: draft[k] for k in ('policy', 'steps', 'batch_size', 'device', 'frames')},
                     operator_next='Проверьте готовый план и подтвердите его запуск на существующей странице обучения.')
        _save_acquisition(db, value, 'acquisition_training_prepared')


def acquisition_training_queued(db, draft, job_id):
    """Called inside the existing Jobs insertion transaction, never a second queue."""
    identifier = draft.get('acquisition_request')
    if identifier:
        row = db.execute('SELECT result FROM cognitive_requests WHERE id=?', (identifier,)).fetchone()
        value = json.loads(row['result']) if row else {}
        if value.get('state') != 'awaiting_training_budget' or value.get('draft_id') != draft['draft_id']:
            raise ValueError('Запрос отменён или подготовленный план изменился')
        value.update(state='training', jobs=[job_id], training_started=True,
                     operator_next='Ход работы и отмена доступны в карточке существующего задания. Роботы освобождены.')
        _save_acquisition(db, value, 'acquisition_training_queued')


def reconcile_acquisition(identifier):
    """Reflect existing jobs and accepted cards; no polling worker or automatic launch."""
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT result FROM cognitive_requests WHERE id=?', (identifier,)).fetchone()
        value = json.loads(row['result'])
        before = store.encoded(value)
        if value.get('jobs') and value['state'] not in ('cancelled', 'rejected', 'accepted'):
            job = db.execute('SELECT * FROM jobs WHERE id=?', (value['jobs'][-1],)).fetchone()
            if job is not None:
                state = job['status']
                if state in ('failed', 'cancelled'):
                    value.update(state='cancelled' if state == 'cancelled' else 'rejected',
                                 operator_next='Обучение ' + ('отменено.' if state == 'cancelled' else 'завершилось ошибкой.') + ' Причина и продолжение — в карточке задания.')
                elif state in ('interrupted', 'unknown'):
                    value['operator_next'] = 'Задание прервано. Продолжите его через существующую очередь; движение не возобновляется.'
                elif state == 'completed':
                    value.update(state='validation', operator_next='Веса готовы. Нужна независимая проверка ROSMASTER без помощи донора.')
                    records = [json.loads(r['body']) for r in db.execute('SELECT body FROM cognitive_skills')]
                    accepted = [r for r in records if r['job_id'] == job['id'] and r['verification']['status'] == 'accepted']
                    exact = [r for r in accepted if r['domain'] == value['domain']]
                    if exact:
                        value.update(state='accepted', skill_id=exact[0]['skill_id'],
                                     operator_next='Навык принят в указанной области. Для продолжения цели нужны свежая сцена, конечная зона и действующее разрешение движения.')
                    elif value['domain'] == 'real' and any(r['domain'] == 'sim' for r in accepted):
                        value.update(state='awaiting_physical_trial',
                                     operator_next='Симуляционная проверка пройдена. Подготовьте отдельное испытание ROSMASTER и подтвердите физический допуск.')
        if store.encoded(value) != before:
            _save_acquisition(db, value, 'acquisition_reconciled')
            if value['state'] in ('accepted', 'cancelled', 'rejected'):
                phase = 'awaiting_resume' if value['state'] == 'accepted' else value['state']
                for linked in db.execute("SELECT id,body FROM cognitive_missions WHERE json_extract(body,'$.acquisition_request')=?", (identifier,)).fetchall():
                    mission = json.loads(linked['body'])
                    mission.update(phase=phase, auto_resume_motion=False, resume_requires_fresh_scene=True)
                    db.execute('UPDATE cognitive_missions SET body=? WHERE id=?', (store.encoded(mission), linked['id']))
    return value
