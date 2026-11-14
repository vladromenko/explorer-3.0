"""Existing Jobs dispatches this finite simulation-only workflow; no trainer."""
import contextlib
import hashlib
import json
import os
import select
import subprocess
import sys
import time
import uuid
from pathlib import Path
from explorer3 import catalog
from explorer3.paths import ROOT,DATA,offline_environment
from explorer3.resources import ResourceLease
from . import broker,store,registry
from .schemas import ObservationEvidence,AssistanceCase,CompatibilityRecord,ExperimentManifest,WorldFact

TASK='fixed_dock';EFFECT='target_observed';RECIPIENT='tank.dock_observed.scripted.v1'
ROLES={
 'MissionCoordinator':'проверенный план и зависимости',
 'SceneObserver':'наблюдения камер с происхождением',
 'AssistanceBroker':'допустимая помощь, G недоступен без обученного артефакта',
 'MemoryCurator':'кандидаты, область применимости и независимые проверки',
 'OutcomeVerifier':'раздельные исходы помощника, получателя и задачи',
 'LearningAndExperimentCurator':'неизменяемый протокол без обучения'}


def update(ident,**changes):
    with catalog.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT body FROM cognitive_missions WHERE id=?',(ident,)).fetchone()
        current=json.loads(row['body'])
        if current['phase'] in ('cancelled','unknown'):raise ValueError('Mission cancelled or interrupted')
        current.update(changes)
        if current['phase'] in ('completed','cancelled','unknown'):
            db.execute('DELETE FROM cognitive_claims WHERE mission_id=?',(ident,))
        db.execute('UPDATE cognitive_missions SET body=? WHERE id=?',(store.encoded(current),ident))
        return current


def tool(m,role,name,args):
    return broker.call_tool({'request_id':uuid.uuid4().hex,'mission_id':m['id'],'generation':m['generation'],
        'decision_id':m['decision_id'],'role':role,'tool':name,'mode':'sim','snapshot_ids':m['snapshot_ids'],
        'expires_at':time.time()+60,'arguments':args})['result']


def observe(robot,payload,source,image=None,ttl=120):
    now=time.time();path=Path(image) if image else None
    return store.put_evidence(ObservationEvidence(id=uuid.uuid4().hex,domain='sim',robot=robot,source=source,
        sample_id=uuid.uuid4().hex,frame_id=robot+'/camera',camera_id=robot+'/camera',
        source_timestamp_ns=int(payload.get('simulation_time_s',0)*1e9),received_timestamp_ns=time.time_ns(),
        clock_domain='mujoco_simulation',created_at=now,expires_at=now+ttl,status='observed',
        artifact_ref=str(path.relative_to(DATA)) if path else None,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        content_sha256=store.digest(payload),payload=payload))


class Sim:
    def __init__(self,work,seed,scene='workshop'):
        self.log=(work/'physics.log').open('w')
        self.process=subprocess.Popen([str(ROOT/'.venv-sim/bin/python'),'-m','explorer3.sim.cooperation',str(work/'scene'),str(seed),scene],
            cwd=ROOT,env={**offline_environment(),'MUJOCO_GL':'egl'},stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
    def request(self,command,timeout=30):
        self.process.stdin.write(json.dumps({'command':command})+'\n');self.process.stdin.flush()
        if not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('Simulation deadline exceeded')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Simulation worker stopped; see physics.log')
        value=json.loads(line)
        if not value['ok']:raise ValueError(value['error'])
        return value['result']
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.log.close()


def model_proposal(work,request):
    source=work/'model-input.json';dest=work/'model-output.json';source.write_text(json.dumps(request,ensure_ascii=False))
    with (work/'model.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv-cognitive/bin/python'),'-m','explorer3.cognitive.model',str(source),str(dest)],
            cwd=ROOT,env=offline_environment(),stdout=log,stderr=subprocess.STDOUT,timeout=100,check=True)
    return json.loads(dest.read_text())


def execute(config,work):
    store.initialize();m=None;sim=None;work=Path(work)
    started=time.monotonic()
    try:
        with ResourceLease('cognitive-mission'):
            sim=Sim(work,config['seed'],config.get('scene','workshop'));snapshot=sim.request('snapshot')
            initial=[]
            for key in ('recipient','donor'):
                camera=snapshot[key]
                item=observe(camera['robot'],{**camera['measurement'],'conditions':snapshot['conditions'],
                    'simulation_time_s':camera['simulation_time_s'],'scene':snapshot['scene'],'model_sha256':snapshot['model_sha256']},'mujoco_robot_rgb',camera['image'])
                initial.append(item.id)
            m=broker.create_mission('ROSMASTER уточняет фиксированную цель, Tank подъезжает и останавливается',initial)
            m=update(m['id'],job_id=work.name,comparison=config['comparison'])
            (work/'mission.json').write_text(json.dumps({'mission_id':m['id']}))
            store.event(m['id'],'role_evidence',{'role':'SceneObserver','evidence_ids':initial})
            memory=tool(m,'AssistanceBroker','retrieve_cases',{'task':TASK,'effect':EFFECT,'recipient_skill_id':RECIPIENT,
                'goal_id':'fixed_dock','comparison':config['comparison']})
            manifest=ExperimentManifest(id=work.name,comparison=config['comparison'],memory_sha256=m['memory_snapshot']['sha256'],
                memory_ids=[x['case']['id'] for x in memory['accepted']],
                skill_versions={c.skill_id:c.version for c in registry.cards().values()},
                scenes=[snapshot['scene']],seeds=[config['seed']],budgets={'model_steps':4,'format_repairs':1,'replans_per_reason':2,
                    'memory_cases':5,'helper_sim_seconds':5,'recipient_sim_seconds':30,'model_wall_seconds':100},
                split='diagnostic',memory_updates=True,evaluator_version=registry.file_hash('explorer3/sim/cooperation.py'),
                scene_sha256=snapshot['model_sha256'],
                model_revision=json.loads((ROOT/'configs/cognitive-model.json').read_text())['revision'],
                model_config_sha256=registry.file_hash('configs/cognitive-model.json'))
            (work/'experiment.json').write_text(store.encoded(manifest))
            store.event(m['id'],'experiment_prepared',{'role':'LearningAndExperimentCurator','manifest':manifest.model_dump(),
                'trainer_started':False,'scientific_hypothesis_tested':False})
            # Memory changes available helper selection, never the motion/safety
            # admission of the recipient. Both helpers still require fresh RGB.
            usable=[x for x in memory['accepted'] if x['case']['actual_effect'].get('direct_pose_verified') is True]
            allowed=['scan_then_dock','wait','human']+(['observe_then_dock'] if usable else [])
            request={'model_config_sha256':manifest.model_config_sha256,'scenario':'S1 fixed shared simulated dock','observations':
                [{'robot':snapshot[k]['robot'],**snapshot[k]['measurement']} for k in ('recipient','donor')],
                'allowed_choices':allowed,'memory':[{'id':x['case']['id'],'status':x['case']['status'],
                    'effect':x['case']['effect'],'actual_effect':x['case']['actual_effect'],
                    'scope_problems':x['scope_problems']} for x in usable],
                'instruction':'Current target visibility is FALSE. Past memory does not make it visible now. A prior camera pose allows choosing observe_then_dock to obtain NEW evidence; otherwise choose scan_then_dock. Neither plan may dock before NEW helper verification.'}
            store.event(m['id'],'model_started',{'role':'MissionCoordinator','allowed_choices':allowed})
            if config['comparison']=='M0':
                proposal={'proposal':{'choice':'scan_then_dock','reason':'Fixed baseline without model or memory'},'backend':'fixed_rules_v1'}
            else:proposal=model_proposal(work,request)
            current=store.mission(m['id'])
            if current['phase']=='cancelled' or current['generation']!=m['generation']:raise ValueError('Late model proposal discarded')
            store.evidence(initial,domain='sim')
            if proposal.get('steps'):broker.record_model_steps(m['id'],m['generation'],proposal['steps'])
            store.event(m['id'],'model_proposal',proposal)
            choice=proposal['proposal']['choice']
            if choice not in allowed:raise ValueError('Model proposal outside current choices')
            if choice in ('wait','human'):
                update(m['id'],phase='completed',outcome='unknown',reason=proposal['proposal']['reason'])
                result={'mission_id':m['id'],'outcome':'unknown','proposal':proposal,'training_started':False}
            else:result=execute_graph(sim,m,initial,snapshot,proposal,memory,usable,config,work)
            result.update(wall_time_s=time.monotonic()-started,physical_robot_access=False,training_started=False)
            (work/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
            return result
    except Exception as exc:
        if m is not None:
            store.event(m['id'],'deterministic_stop',{'error':str(exc),'fallback':'hold; no blind replan or resume'})
            if store.mission(m['id'])['phase'] not in ('cancelled','unknown'):
                update(m['id'],phase='completed',outcome='unknown',error=str(exc))
        raise
    finally:
        if sim is not None:sim.close()


def execute_graph(sim,m,initial,snapshot,proposal,memory,usable,config,work):
    direct=proposal['proposal']['choice']=='observe_then_dock'
    helper=registry.get('rosmaster.'+('observe_dock' if direct else 'scan_dock')+'.geometric.v1')
    recipient=registry.get(RECIPIENT)
    manifest=ExperimentManifest.model_validate_json((work/'experiment.json').read_text())
    for card in (helper,recipient):
        if manifest.skill_versions.get(card.skill_id)!=card.version:raise ValueError('Skill changed after experiment preparation')
    graph={'nodes':[
        {'id':'observe','skill_id':helper.skill_id,'skill_version':helper.version,'robot':'rosmaster',
            'goal_id':'fixed_dock_marker','deadline_s':helper.timeout_s,'depends_on':[]},
        {'id':'dock','skill_id':recipient.skill_id,'skill_version':recipient.version,'robot':'tank',
            'goal_id':'fixed_dock','deadline_s':30,'depends_on':['observe']}],
        'evidence_ids':initial,'rationale':proposal['proposal']['reason']}
    tool(m,'MissionCoordinator','propose_graph',graph)
    tool(m,'MissionCoordinator','prepare_skill_job',{'node_id':'observe'})
    with broker.execution_step(m['id'],'observe'):measured=sim.request('observe' if direct else 'scan')
    if not measured['recipient_held']:raise ValueError('Recipient moved before admission')
    helper_e=observe('rosmaster',measured,'mujoco_robot_rgb',measured['image'])
    if measured['helper_outcome']!='success':
        m=broker.refresh_snapshot(m['id'],[helper_e.id])
        scope=CompatibilityRecord(recipient_skill_id=recipient.skill_id,recipient_version=recipient.version,
            policy_sha256=recipient.policy_sha256,camera_sha256=recipient.camera_sha256,calibration_sha256=None,
            domain='sim',goal_id='fixed_dock',checked_conditions=['fixed_scene','recipient_stationary'],
            unknown_conditions=['target_observed_by_helper'],evidence_ids=[helper_e.id])
        failed=AssistanceCase(id=uuid.uuid4().hex,task=TASK,effect=EFFECT,domain='sim',donor='rosmaster',recipient='tank',
            donor_episode=work.name+'-observe',recipient_episode=work.name+'-not-started',
            reason='Helper camera did not establish target visibility; recipient remained held',before_evidence_ids=m['snapshot_ids'],
            compatibility=scope,assistance_type='observation',expected_effect='Fresh visible target before docking',
            actions_provenance='geometric',actual_effect={'direct_pose_verified':False,'scene':snapshot['scene'],
                'model_sha256':snapshot['model_sha256'],'observation_evidence_id':helper_e.id})
        tool(m,'MemoryCurator','add_memory_candidate',failed.model_dump())
        evidence=observe('tank',{'helper':'failure','recipient':'unknown','task':'unknown','episode':work.name,
            'compatibility_sha256':store.digest(scope),'helper_evidence_id':helper_e.id,'recipient_started':False},'fixed_outcome_verifier')
        store.verify_case(failed.id,work.name,helper='failure',recipient='unknown',task='unknown',evidence_ids=[evidence.id],
            verifier_version='shared-dock-v1',mission_id=m['id'],generation=m['generation'])
        raise ValueError('Helper did not establish target visibility')
    # Source is explicitly the donor's measured RGB; no invented Tank camera.
    fact=store.put_fact(WorldFact(id=uuid.uuid4().hex,subject='fixed_dock_marker',relation='visible_in_camera',
        value={'visible':True,'boxes':measured['measurement']['detections']},domain='sim',frame_id=helper_e.frame_id,
        evidence_ids=[helper_e.id],valid_since=helper_e.created_at,expires_at=helper_e.expires_at,status='observed'))
    recipient_e=observe('tank',{'conditions':{**measured['conditions'],'target_observed_by_helper':fact.value['visible']},
        'world_fact_id':fact.id,'donor_evidence_id':helper_e.id,
        'simulation_time_s':measured['simulation_time_s'],'observation_transferred':True},'verified_helper_observation')
    update(m['id'],verified_nodes=['observe'])
    m=broker.refresh_snapshot(m['id'],[helper_e.id,recipient_e.id])
    store.event(m['id'],'helper_verified',{'role':'OutcomeVerifier','evidence_ids':m['snapshot_ids'],
        'helper':'success','recipient':'unknown','task':'unknown'})
    tool(m,'MissionCoordinator','prepare_skill_job',{'node_id':'dock'})
    with broker.execution_step(m['id'],'dock'):outcomes=sim.request('dock')
    scope=CompatibilityRecord(recipient_skill_id=recipient.skill_id,recipient_version=recipient.version,
        policy_sha256=recipient.policy_sha256,camera_sha256=recipient.camera_sha256,calibration_sha256=None,
        domain='sim',goal_id='fixed_dock',checked_conditions=recipient.preconditions,unknown_conditions=[],evidence_ids=initial)
    case=AssistanceCase(id=uuid.uuid4().hex,task=TASK,effect=EFFECT,domain='sim',donor='rosmaster',recipient='tank',
        donor_episode=work.name+'-observe',recipient_episode=work.name+'-dock',reason='Fresh robot camera observation enables fixed-goal docking',
        before_evidence_ids=m['snapshot_ids'],compatibility=scope,assistance_type='observation',
        expected_effect='Target seen by helper; recipient reaches dock and stops',actions_provenance='geometric',
        actual_effect={'direct_pose_verified':True,'scene':snapshot['scene'],'model_sha256':snapshot['model_sha256'],
            'observed_pixels':measured['measurement']['detections'],'helper_skill_version':helper.version},
        delay_s=measured['simulation_time_s'],operator_active_s=0.)
    # Diagnostic memory updates only; independent outcome never comes from LLM.
    tool(m,'MemoryCurator','add_memory_candidate',case.model_dump())
    verifier_ids=[]
    for candidate in [case,*[AssistanceCase.model_validate(x['case']) for x in usable[:1]
                            if not store.compatibility(x['case'],recipient,'fixed_dock','sim',{'scene':snapshot['scene'],'model_sha256':snapshot['model_sha256']})]]:
        payload={**outcomes,'episode':work.name,'compatibility_sha256':store.digest(candidate.compatibility)}
        evidence=observe('tank',payload,'fixed_outcome_verifier')
        store.verify_case(candidate.id,work.name,helper=outcomes['helper'],recipient=outcomes['recipient'],task=outcomes['task'],
            evidence_ids=[evidence.id],verifier_version=outcomes['version'],mission_id=m['id'],generation=m['generation'])
        verifier_ids.append(evidence.id)
    manifest=ExperimentManifest.model_validate_json((work/'experiment.json').read_text())
    store.event(m['id'],'outcomes_verified',{'role':'OutcomeVerifier','measurements':outcomes,'evidence_ids':verifier_ids})
    update(m['id'],phase='completed',outcome=outcomes['task'],verified_nodes=['observe','dock'],case_id=case.id)
    return {'mission_id':m['id'],'outcome':outcomes['task'],'helper':outcomes['helper'],'recipient':outcomes['recipient'],
        'task':outcomes['task'],'proposal':proposal,'graph':graph,'memory':memory,'case_id':case.id,
        'evidence_ids':initial+m['snapshot_ids']+verifier_ids,'experiment':manifest.model_dump(),
        'scientific_hypothesis_tested':False,'limitations':['privileged scripted docking teacher','fixed nominal shared scene','uncalibrated hardware model']}


if __name__=='__main__':execute(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
