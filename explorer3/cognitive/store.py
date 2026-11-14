"""Versioned local evidence and assistance memory in Explorer's existing SQLite."""
import hashlib
import contextlib
import json
import time
import uuid
from explorer3 import catalog
from explorer3.paths import DATA
from .schemas import ObservationEvidence,WorldFact,AssistanceCase


def encoded(value):
    if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()


def initialize():
    with catalog.connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS cognitive_migrations(version INTEGER PRIMARY KEY, applied REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_evidence(id TEXT PRIMARY KEY, body TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_facts(id TEXT PRIMARY KEY, body TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_cases(id TEXT PRIMARY KEY, body TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS cognitive_case_search USING fts5(id UNINDEXED, task, effect, reason);
        CREATE TABLE IF NOT EXISTS cognitive_verifications(id TEXT PRIMARY KEY, case_id TEXT NOT NULL, episode TEXT NOT NULL,
            body TEXT NOT NULL, UNIQUE(case_id,episode));
        CREATE TABLE IF NOT EXISTS cognitive_missions(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_requests(id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, decision_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL, state TEXT NOT NULL, result TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_events(id INTEGER PRIMARY KEY, mission_id TEXT, kind TEXT NOT NULL,
            body TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS cognitive_claims(resource TEXT PRIMARY KEY, mission_id TEXT NOT NULL,
            generation INTEGER NOT NULL, expires REAL NOT NULL);
        ''')
        db.execute('INSERT OR IGNORE INTO cognitive_migrations VALUES(1,?)',(time.time(),))


def event(mission_id,kind,value,db=None):
    if db is None:
        with catalog.connect() as connection:event(mission_id,kind,value,connection)
    else:db.execute('INSERT INTO cognitive_events(mission_id,kind,body,created) VALUES(?,?,?,?)',
                   (mission_id,kind,encoded(value),time.time()))


def validate_evidence_artifact(value):
    item=ObservationEvidence.model_validate(value)
    if digest(item.payload)!=item.content_sha256:raise ValueError('Evidence payload hash mismatch')
    if item.artifact_ref:
        path=(DATA/item.artifact_ref).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():raise ValueError('Evidence artifact unavailable')
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):sha.update(block)
        if sha.hexdigest()!=item.artifact_sha256:raise ValueError('Evidence artifact hash mismatch')
    return item


def put_evidence(value):
    item=validate_evidence_artifact(value)
    body=encoded(item)
    with catalog.connect() as db:
        old=db.execute('SELECT body FROM cognitive_evidence WHERE id=?',(item.id,)).fetchone()
        if old and old['body']!=body:raise ValueError('Evidence is immutable')
        db.execute('INSERT OR IGNORE INTO cognitive_evidence VALUES(?,?,?)',(item.id,body,item.expires_at))
    return item


def evidence(ids,*,domain=None,fresh=True,now=None):
    now=time.time() if now is None else now;items=[]
    with catalog.connect() as db:
        for ident in ids:
            row=db.execute('SELECT body FROM cognitive_evidence WHERE id=?',(ident,)).fetchone()
            if row is None:raise ValueError('Unknown evidence: '+ident)
            item=ObservationEvidence.model_validate_json(row['body'])
            if fresh and (item.expires_at<=now or item.created_at>now+.1):raise ValueError('Stale or future evidence: '+ident)
            if domain and item.domain!=domain:raise ValueError('Evidence domain mismatch')
            items.append(item)
    return items


def put_fact(value):
    fact=WorldFact.model_validate(value);items=evidence(fact.evidence_ids,domain=fact.domain)
    if fact.domain=='real' and fact.frame_id in ('world','map'):raise ValueError('Shared real calibration unavailable')
    if fact.expires_at>min(e.expires_at for e in items):raise ValueError('Fact cannot outlive its evidence')
    body=encoded(fact)
    with catalog.connect() as db:
        old=db.execute('SELECT body FROM cognitive_facts WHERE id=?',(fact.id,)).fetchone()
        if old and old['body']!=body:raise ValueError('Keep contradictory facts as separate records')
        db.execute('INSERT OR IGNORE INTO cognitive_facts VALUES(?,?,?)',(fact.id,body,fact.expires_at))
    return fact


def add_case(value,db=None):
    item=AssistanceCase.model_validate(value)
    evidence(item.before_evidence_ids,domain=item.domain,fresh=False)
    # Model proposals can add candidates, never their own measured success.
    if item.status!='candidate' or any(x!='unknown' for x in (item.helper_outcome,item.recipient_outcome,item.task_outcome)):
        raise ValueError('Only the independent verifier can assign outcomes')
    if item.compatibility.domain!=item.domain:raise ValueError('Memory scope domain mismatch')
    with (catalog.connect() if db is None else contextlib.nullcontext(db)) as db:
        old=db.execute('SELECT body FROM cognitive_cases WHERE id=?',(item.id,)).fetchone()
        if old and old['body']!=encoded(item):raise ValueError('Memory candidate identity collision')
        if old is None:
            db.execute('INSERT INTO cognitive_cases VALUES(?,?,?,?)',(item.id,encoded(item),'candidate',time.time()))
            db.execute('INSERT INTO cognitive_case_search VALUES(?,?,?,?)',(item.id,item.task,item.effect,item.reason))
            event(None,'memory_candidate',item,db)
    return item.id


def verify_case(case_id,episode,*,helper,recipient,task,evidence_ids,verifier_version,mission_id=None,generation=None):
    """Called by the deterministic outcome path; deliberately not an agent tool."""
    values=(helper,recipient,task)
    if any(x not in ('success','failure','unknown','cancelled','hardware_error') for x in values):raise ValueError('Unknown outcome')
    items=evidence(evidence_ids,fresh=False)
    if not items or any(e.source!='fixed_outcome_verifier' for e in items):raise ValueError('Independent outcome evidence required')
    payload={'helper':helper,'recipient':recipient,'task':task,'evidence_ids':evidence_ids,'verifier_version':verifier_version}
    if not any(all(e.payload.get(k)==v for k,v in zip(('helper','recipient','task'),values)) for e in items):
        raise ValueError('Outcomes disagree with verifier measurements')
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if mission_id is not None:
            current=json.loads(db.execute('SELECT body FROM cognitive_missions WHERE id=?',(mission_id,)).fetchone()['body'])
            if current['phase'] in ('cancelled','unknown','completed') or current['generation']!=generation:
                raise ValueError('Late verification discarded')
        row=db.execute('SELECT * FROM cognitive_cases WHERE id=?',(case_id,)).fetchone()
        if row is None:raise ValueError('Unknown case')
        case=AssistanceCase.model_validate_json(row['body'])
        if any(e.domain!=case.domain for e in items):raise ValueError('Outcome domain mismatch')
        if not any(e.payload.get('episode')==episode and e.payload.get('compatibility_sha256')==digest(case.compatibility) for e in items):
            raise ValueError('Verification episode or recipient scope mismatch')
        old=db.execute('SELECT body FROM cognitive_verifications WHERE case_id=? AND episode=?',(case_id,episode)).fetchone()
        if old and old['body']!=encoded(payload):raise ValueError('Verification is immutable')
        db.execute('INSERT OR IGNORE INTO cognitive_verifications VALUES(?,?,?,?)',
                   (uuid.uuid4().hex,case_id,episode,encoded(payload),))
        rows=[json.loads(r['body']) for r in db.execute('SELECT body FROM cognitive_verifications WHERE case_id=?',(case_id,))]
        positive=sum(r['helper']=='success' and r['recipient']=='success' and r['task']=='success' for r in rows)
        negative=any(r['helper']=='failure' or r['recipient']=='failure' or r['task']=='failure' for r in rows)
        status='contradicted' if negative else 'supported_in_scope' if positive>=2 else 'candidate'
        updated=case.model_copy(update={'status':status,'helper_outcome':helper,'recipient_outcome':recipient,'task_outcome':task})
        db.execute('UPDATE cognitive_cases SET status=?,body=? WHERE id=?',(status,encoded(updated),case_id))
        if old is None:event(None,'memory_verified',{'case_id':case_id,'episode':episode,'verification':payload,'status':status},db)
    return status


def cases():
    with catalog.connect() as db:return [json.loads(r['body']) for r in db.execute('SELECT body FROM cognitive_cases ORDER BY created DESC')]


def compatibility(case,skill,goal,domain,context=None):
    scope=case['compatibility'];problems=[]
    expected={'recipient_skill_id':skill.skill_id,'recipient_version':skill.version,
        'policy_sha256':skill.policy_sha256,'camera_sha256':skill.camera_sha256,
        'calibration_sha256':skill.calibration_sha256,'goal_id':goal,'domain':domain}
    for key,value in expected.items():
        if scope.get(key)!=value:problems.append(key+' changed')
    for key,value in (context or {}).items():
        if case.get('actual_effect',{}).get(key)!=value:problems.append(key+' changed')
    if scope.get('unknown_conditions'):problems.append('Continuation conditions unknown')
    if case['status'] in ('contradicted','stale'):problems.append('Memory is '+case['status'])
    return problems


def retrieve(task,effect,skill,goal,domain,*,comparison='M3',limit=5,snapshot=None,context=None):
    if comparison not in ('M0','M1','M2','M3'):raise ValueError('Unknown comparison')
    if not 1<=limit<=5:raise ValueError('At most five memory cases per decision')
    if comparison in ('M0','M1'):return {'accepted':[],'rejected':[],'memory_enabled':False}
    query=' OR '.join('"'+x.replace('"','""')+'"' for x in (task,effect))
    frozen={c['id']:c for c in snapshot['cases']} if snapshot is not None else None
    with catalog.connect() as db:
        rows=db.execute('''SELECT c.body FROM cognitive_case_search f JOIN cognitive_cases c ON c.id=f.id
            WHERE cognitive_case_search MATCH ? AND (? IS NULL OR c.id IN (SELECT value FROM json_each(?)))
            ORDER BY bm25(cognitive_case_search),c.created DESC LIMIT 40''',
            (query,None if frozen is None else 1,encoded(list(frozen or {})))).fetchall()
    accepted=[];rejected=[]
    for row in rows:
        current=json.loads(row['body']);case=frozen[current['id']] if frozen is not None else current
        if case['task']==task and case['effect']==effect and case['domain']==domain:
            problems=compatibility(case,skill,goal,domain,context)
            hard=case['assistance_type'] not in ('observation','recovery','human','wait')
            if hard or (comparison=='M3' and problems):
                rejected.append({'id':case['id'],'reasons':problems or ['Unsupported assistance']})
            elif len(accepted)<limit:
                accepted.append({'case':case,'compatibility_checked':comparison=='M3',
                    'empirically_supported':case['status']=='supported_in_scope','scope_problems':problems})
    return {'accepted':accepted,'rejected':rejected[:10],'memory_enabled':True}


def memory_snapshot():
    value=cases()
    return {'cases':value,'sha256':digest(value)}


def mission(identifier):
    with catalog.connect() as db:row=db.execute('SELECT body FROM cognitive_missions WHERE id=?',(identifier,)).fetchone()
    if row is None:raise ValueError('Unknown mission')
    return json.loads(row['body'])


def events(identifier):
    with catalog.connect() as db:
        return [{**dict(r),'body':json.loads(r['body'])} for r in db.execute(
            'SELECT * FROM cognitive_events WHERE mission_id=? ORDER BY id',(identifier,))]


def recover_interrupted():
    """Application startup: unknown outcomes require reconciliation, never replay."""
    with catalog.connect() as db:
        db.execute("UPDATE cognitive_requests SET state='unknown' WHERE state='running'")
        for row in db.execute('SELECT * FROM cognitive_missions').fetchall():
            value=json.loads(row['body'])
            if value['phase'] in ('planning','planned','running'):
                value.update(phase='unknown',generation=value['generation']+1,error='Application restarted; reconcile before a new mission')
                db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(encoded(value),row['id']))
                event(row['id'],'restart_unknown',{'generation':value['generation']},db)
        db.execute('DELETE FROM cognitive_claims')
