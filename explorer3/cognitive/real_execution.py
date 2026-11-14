"""Finite real TaskGraphs reuse existing trial executors and operator outcomes.

No worker thread, motor API, policy implementation or automatic retry lives here.
The UI supplies heartbeats; unknown outcome blocks all dependent steps.
"""
import json
import threading
import time
import uuid
from explorer3 import catalog,trials
from explorer3.paths import DATA
from explorer3.resources import ResourceLease
from explorer3.training_drafts import save_content
from . import store,registry
from .schemas import TaskGraph,TaskNode


def prepare(trial_ids,evidence_ids,task,*,memory_case_id=None):
    if not 1<=len(trial_ids)<=4 or len(set(trial_ids))!=len(trial_ids):raise ValueError('Выберите от одного до четырёх разных испытаний')
    if not isinstance(task,str) or not 1<=len(task)<=400:raise ValueError('Нужно короткое описание задачи')
    evidence=store.evidence(evidence_ids,domain='real',fresh=False)
    for item in evidence:store.validate_evidence_artifact(item)
    nodes=[];plans={}
    for index,ident in enumerate(trial_ids):
        plan=trials.load(ident);card=registry.get('trial.'+ident)
        if plan['kind']=='policy':trials.verify_policy_artifacts(plan)
        node=TaskNode(id='step-'+str(index+1),skill_id=card.skill_id,skill_version=card.version,
            robot=card.robot,goal_id=card.goals[0],depends_on=[nodes[-1].id] if nodes else [],
            deadline_s=min(60.,card.timeout_s))
        nodes.append(node);plans[node.id]={'trial_id':ident,'sha256':trials.plan_hash(plan),'profile':plan['profile'],'kind':plan['kind']}
    graph=TaskGraph(nodes=nodes,evidence_ids=evidence_ids,rationale=task)
    if not {n.robot for n in nodes}<={e.robot for e in evidence}:raise ValueError('Нужны исходные наблюдения каждого выбранного робота')
    registry.validate_graph(graph,'real',shadow=True)
    value={'schema':'real_task_graph_v1','graph':graph.model_dump(),'trials':plans,
        'created_at':time.time(),'mode':'real','permission_granted':False,
        'verifier':'operator outcome checked against finalized original episode',
        'source_evidence_hashes':{e.id:e.content_sha256 for e in evidence},
        'execution_started':False,'training_started':False}
    if memory_case_id:
        from .transfer import prepare as prepare_transfer
        value['memory_transfer']=prepare_transfer(memory_case_id,nodes[-1].skill_id,evidence_ids)
    return save_content(DATA/'cognitive/real-graphs',value),value


def load(ident):
    import hashlib
    if len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid graph ID')
    raw=(DATA/'cognitive/real-graphs'/(ident+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ident:raise ValueError('Graph changed')
    value=json.loads(raw);graph=TaskGraph.model_validate(value['graph'])
    registry.validate_graph(graph,'real',shadow=True)
    for entry in value['trials'].values():
        if trials.plan_hash(trials.load(entry['trial_id']))!=entry['sha256']:raise ValueError('Trial changed')
    for item in store.evidence(graph.evidence_ids,domain='real',fresh=False):
        store.validate_evidence_artifact(item)
        if item.content_sha256!=value['source_evidence_hashes'].get(item.id):raise ValueError('Source evidence changed')
    if value.get('memory_transfer'):
        from .transfer import validate
        validate(value['memory_transfer'])
    return value


def propose_order(config,work):
    from .run import model_proposal
    from .model import TrialOrder
    value=load(config['plan_id']);graph=value['graph']
    sources=store.evidence(graph['evidence_ids'],domain='real',fresh=False)
    for item in sources:
        store.validate_evidence_artifact(item)
        if item.content_sha256!=value['source_evidence_hashes'].get(item.id):raise ValueError('Source evidence changed')
    skills=[]
    for node in graph['nodes']:
        card=registry.get(node['skill_id']).model_dump()
        skills.append({'trial_id':value['trials'][node['id']]['trial_id'],
            'card':{key:card[key] for key in ('skill_id','version','robot','kind','profile','goals','preconditions','postconditions','limitation')}})
    request={'purpose':'trial_order','task':graph['rationale'],
        'model_config_sha256':registry.file_hash('configs/cognitive-model.json'),
        'skills':skills,
        'observations':[{'id':e.id,'robot':e.robot,'source':e.source,'frame_id':e.frame_id,
            'created_at':e.created_at,'expires_at':e.expires_at,'status':e.status,'archived':e.payload.get('archived',False)} for e in sources],
        'motion_authorized':False,'training_authorized':False,'memory_hypothesis':value.get('memory_transfer')}
    result=model_proposal(work,request)
    proposal=TrialOrder.model_validate(result['proposal'])
    if not set(proposal.trial_ids)<={v['trial_id'] for v in value['trials'].values()}:raise ValueError('Unknown proposed trial')
    result.update(source_plan_id=config['plan_id'],execution_started=False,training_started=False,physical_access=False)
    (work/'result.json').write_text(store.encoded(result));return result


def apply_order(plan_id,result):
    """A proposal creates another reviewable plan, never starts a mission."""
    from .model import TrialOrder
    value=load(plan_id);proposal=TrialOrder.model_validate(result['proposal'])
    if result.get('source_plan_id')!=plan_id:raise ValueError('Proposal belongs to another plan')
    if proposal.blocked:raise ValueError(proposal.reason)
    if not set(proposal.trial_ids)<={v['trial_id'] for v in value['trials'].values()}:raise ValueError('Unknown proposed trial')
    return prepare(proposal.trial_ids,value['graph']['evidence_ids'],value['graph']['rationale'],
        memory_case_id=value.get('memory_transfer',{}).get('source_case_id'))


class Executor:
    def __init__(self,adapter):
        self.adapter=adapter;self.lock=threading.RLock();self.claim=None;self.state=None;self.starting=False;self.last_seen=0.;self.cancel_epoch=0

    def save(self,kind):
        with catalog.connect() as db:
            db.execute('INSERT OR REPLACE INTO cognitive_missions VALUES(?,?)',(self.state['id'],store.encoded(self.state)))
            store.event(self.state['id'],kind,self.state,db)

    def start(self,plan_id,*,operator_confirmed=False):
        with self.lock:
            if not operator_confirmed:raise PermissionError('Требуется подтверждение конкретной физической миссии')
            if self.claim or self.starting:raise ValueError('Mission already active or finishing its start')
            self.starting=True;epoch=self.cancel_epoch;self.last_seen=time.monotonic()
        try:
            plan=load(plan_id)
            # Trials remain separately admitted: no stale/missing permission is
            # silently created by a model or by this dispatcher.
            for entry in plan['trials'].values():trials.require_physical(trials.load(entry['trial_id']))
            with self.lock:
                if epoch!=self.cancel_epoch or time.monotonic()-self.last_seen>2:raise ValueError('Start cancelled or browser lost')
                claim=ResourceLease('cognitive-mission');claim.__enter__();self.claim=claim
                self.state={'id':uuid.uuid4().hex,'domain':'real','phase':'running','generation':0,
                    'plan_id':plan_id,'plan':plan,'index':0,'verified_nodes':[],'node_outcomes':[],
                    'created_at':time.time(),'expires_at':time.time()+600,
                    'stage':'starting','task_outcome':'unknown','training_started':False}
                self.save('real_mission_started');self.starting=False
            self._next()
        except Exception:
            self.cancel('Start failed; reconcile physical state');raise
        finally:
            with self.lock:self.starting=False
        return self.state['id']

    def heartbeat(self):
        self.last_seen=time.monotonic()
        if self.claim:self.adapter.heartbeat()

    def _next(self):
        with self.lock:
            if not self.claim or self.starting:raise ValueError('No admitted start')
            if time.time()>=self.state['expires_at'] or time.monotonic()-self.last_seen>2:raise ValueError('Mission permission expired')
            value=self.state['plan'];nodes=value['graph']['nodes'];index=self.state['index']
            if index==len(nodes):
                self.state.update(phase='completed',task_outcome='success',stage='done');self.save('real_mission_completed')
                try:
                    if value.get('memory_transfer'):
                        from .transfer import record_continuation
                        case=record_continuation(value['memory_transfer'],self.state['node_outcomes'][-1]['episode_id'],mission_id=self.state['id'])
                        self.state['transfer_case_id']=case;self.save('real_transfer_linked')
                except Exception as exc:
                    self.state['memory_error']=str(exc);self.save('real_transfer_record_failed')
                finally:self.claim.__exit__(None,None,None);self.claim=None
                return
            node=nodes[index]
            if not set(node['depends_on'])<=set(self.state['verified_nodes']):raise ValueError('Unverified predecessor')
            entry=value['trials'][node['id']]
            self.state.update(node_id=node['id'],node_started_at=time.monotonic(),node_deadline_s=node.get('deadline_s',60.),stage='starting')
            self.save('real_step_started');generation=self.state['generation'];self.starting=True
        try:
            load(self.state['plan_id']);trials.require_physical(trials.load(entry['trial_id']))
            def admitted():
                return bool(self.claim and generation==self.state['generation'] and time.time()<self.state['expires_at'] and time.monotonic()-self.last_seen<=2)
            if not admitted():raise ValueError('Start cancelled')
            self.adapter.start(entry['trial_id'],self.state['id'],admitted)
            with self.lock:
                if not self.claim or generation!=self.state['generation']:raise ValueError('Start cancelled')
                self.state['stage']='executing';self.save('real_step_executing')
        finally:
            with self.lock:self.starting=False

    def finish_step(self,outcome):
        if outcome not in ('success','failure'):raise ValueError('Explicit measured/operator outcome required')
        with self.lock:
            if not self.claim or self.state['stage']!='executing':raise ValueError('No executing step')
            self.adapter.finish(outcome)
            self.state.update(stage='saving',requested_outcome=outcome);self.save('operator_outcome_submitted')

    def tick(self):
        with self.lock:
            if not self.claim:return
            current=store.mission(self.state['id'])
            if current['generation']!=self.state['generation'] or current['phase']!='running':
                self.cancel('Mission invalidated');return
            if time.time()>=self.state['expires_at'] or time.monotonic()-self.last_seen>2:
                self.cancel('Browser lease expired');return
            if self.starting:return
            if self.state['stage']=='ready_for_next':return
            status=self.adapter.status()
            if status['active']:
                if time.monotonic()-self.state['node_started_at']>self.state['node_deadline_s']:self.cancel('Step deadline exceeded')
                return
            if self.state['stage']!='saving' or status.get('error') or status.get('outcome')!=self.state.get('requested_outcome'):
                self.cancel(status.get('error') or 'Unknown outcome; no automatic continuation');return
            episode_id=status.get('episode_id')
            row=next((e for e in catalog.rows('episodes') if e['id']==episode_id),None)
            expected=self.state['plan']['trials'][self.state['node_id']]['profile']
            if row is None or row['profile']!=expected or row['outcome']!=status['outcome'] or row['frames']<1:
                self.cancel('Finalized episode does not confirm submitted outcome');return
            self.adapter.close_step()
            self.state['node_outcomes'].append({'node':self.state['node_id'],'episode_id':episode_id,
                'outcome':row['outcome'],'source':'operator_label_and_saved_episode','independent_automatic_measurement':False})
            if row['outcome']!='success':
                self.state['task_outcome']='failure';self.cancel('Step failed')
                transfer=self.state['plan'].get('memory_transfer')
                if transfer and transfer['recipient_skill']['profile']==row['profile']:
                    try:
                        from .transfer import record_continuation
                        record_continuation(transfer,episode_id,mission_id=self.state['id'])
                    except Exception as exc:
                        self.state['memory_error']=str(exc);self.save('real_transfer_record_failed')
                return
            self.state['verified_nodes'].append(self.state['node_id']);self.state['index']+=1
            # No automatic movement into another robot: recheck scene/lease on
            # the explicit advance button, rather than treating file save as START.
            self.state['stage']='ready_for_next';self.save('real_step_verified')
            if self.state['index']==len(self.state['plan']['graph']['nodes']):self._next()

    def advance(self):
        with self.lock:
            if not self.claim or self.state['stage']!='ready_for_next':raise ValueError('Previous step not verified')
            if time.monotonic()-self.last_seen>2:raise ValueError('Browser lease expired')
        try:self._next()
        except Exception:
            self.cancel('Next step could not start');raise

    def cancel(self,reason='STOP'):
        with self.lock:
            self.cancel_epoch+=1
            if not self.claim:return
            claim=self.claim;self.claim=None
            self.state.update(phase='cancelled',generation=self.state['generation']+1,stage='stopped',reason=reason)
            self.save('real_mission_cancelled')
        try:self.adapter.stop(reason)
        finally:claim.__exit__(None,None,None)
