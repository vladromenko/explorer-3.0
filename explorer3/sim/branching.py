"""R5 protocol adaptation: matched simulator/policy branches, no optimization.

This is our finite experimental protocol, not a reproduction of a paper's code.
Simulator truth is only used by the explicitly privileged correction expert and
outcome scoring. Diagnostic hold is not a learned policy or an intervention model.
"""
import argparse
import copy
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
import numpy as np
from . import state_codec
from explorer3.branching_config import validate_config
from .factory import make,EXPERTS
from .mobile import MobileDock
from explorer3.act_client import PolicyClient
from explorer3.profiles import profile as profile_config


class DiagnosticHold:
    metadata={'kind':'diagnostic_hold','learned':False,'privileged_input':False}
    def __init__(self,env):self.env=env;self.calls=0
    def action(self):self.calls+=1;return hold(self.env)
    def snapshot(self):return {'calls':self.calls}
    def restore(self,state):self.calls=state['calls']
    def reset(self):self.calls=0
    def close(self):pass


class FrozenPolicy:
    def __init__(self,env,checkpoint,log,device):
        self.env=env;self.cfg=profile_config(env.profile)
        self.client=PolicyClient(Path(checkpoint),Path(log),device=device,response_timeout=60)
        self.metadata={**self.client.ready,'kind':'frozen_policy','learned':True,
                       'privileged_input':False,'checkpoint':str(checkpoint),'device':device}
    def action(self):
        obs={'state':self.env.observation()['state'],'rgb':self.env.render()}
        return self.client.action(obs,self.cfg)
    def snapshot(self):return self.client.request('snapshot')
    def restore(self,state):self.client.request('restore',snapshot=state)
    def reset(self):self.client.request('reset')
    def close(self):self.client.close()


def hold(env):
    return np.zeros(env.action_space.shape) if isinstance(env,MobileDock) else env.observation()['state'].astype(float).copy()


class RelativeClock:
    """Existing geometric expert starts its finite sequence at assistance arrival."""
    def __init__(self,env):self.env=env;self.start=env.elapsed
    @property
    def elapsed(self):return self.env.elapsed-self.start
    def __getattr__(self,name):return getattr(self.env,name)




def experiment(env,policy,output,config):
    """Run one lineage. Caller owns the compute lease and any policy subprocess."""
    validate_config(config)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # No silent overwriting of scientific evidence.
    if (output/'snapshot.json').exists():raise ValueError('Branch directory already contains a snapshot')
    from explorer3.gain_preparation import sample,observation
    decision_history=[]
    for _ in range(math.ceil(config['warmup_s']/env.dt-1e-10)):
        action=policy.action()
        decision_history.append(sample(env.profile,env.observation()['state'],action,env.elapsed))
        _,_,terminated,truncated,_=env.step(action)
        if terminated or truncated:raise ValueError('Task ended before the requested branch point')
    point={'environment':env.snapshot(),'policy':policy.snapshot(),'policy_identity':policy.metadata}
    lineage=uuid.uuid4().hex
    digest=state_codec.dump(output/'snapshot.json',point)
    # Reload the on-disk snapshot, not only an in-memory object.
    point=state_codec.load(output/'snapshot.json')
    wait_steps=math.ceil(config['delay_s']/env.dt-1e-10)
    help_steps=math.ceil(config['correction_s']/env.dt-1e-10)
    horizon_steps=math.ceil(config['horizon_s']/env.dt-1e-10)
    intervention_end=wait_steps+help_steps
    if intervention_end>horizon_steps:raise ValueError('Rounded intervention exceeds horizon')
    decision_context={'features':observation(decision_history,env.elapsed,
        env.action_space.high-env.action_space.low,wait_steps*env.dt),
        'source':'controller_visible_history','domain':'sim','truth_features':False,
        'decision_sim_s':env.elapsed,'measured_hardware':False}
    results=[];replay={};segments=[]
    started=time.monotonic()
    for repeat in range(config['repeats']):
        for branch in ('continue','hold_only','correction'):
            env.restore(point['environment']);policy.restore(point['policy'])
            initial=state_codec.dump(output/f'{branch}-{repeat}-initial.json',
                {'environment':env.snapshot(),'policy':policy.snapshot(),'policy_identity':policy.metadata})
            if initial!=digest:raise RuntimeError('Full state did not restore exactly before branching')
            rows=[];expert=None;reset_step=None;capture=None
            reason='post_decision_horizon';outcome=None
            for step in range(horizon_steps):
                phase='policy';source=policy.metadata['kind'];inference_s=0.
                if branch!='continue' and step<intervention_end:
                    if branch=='correction' and step>=wait_steps:
                        if expert is None:expert=EXPERTS[env.profile](RelativeClock(env))
                        action=expert.action();phase='correction';source='scripted_expert'
                        if repeat==0 and config.get('capture_correction',False):
                            if capture is None:
                                from .capture import SimCapture
                                capture=SimCapture(env,config['seed'],'scripted_expert')
                                capture.meta.update(lineage=lineage,session=lineage,branch='correction',
                                    snapshot_sha256=digest,segment_role='bounded_correction',
                                    training_eligible=False,expert_privileged=True)
                            capture.add(env,action)
                    else:action=hold(env);phase='waiting' if step<wait_steps else 'matched_hold';source='safety'
                else:
                    if branch!='continue' and reset_step is None:
                        # Both intervention arms clear chunks/observation queues at
                        # the SAME return boundary; continuation preserves its queue.
                        policy.reset();reset_step=step
                    tick=time.monotonic();action=policy.action();inference_s=time.monotonic()-tick
                action=np.asarray(action,dtype=float)
                if action.shape!=env.action_space.shape or not np.isfinite(action).all():
                    raise ValueError('Invalid policy action')
                before=env.elapsed
                _,_,terminated,truncated,info=env.step(action)
                rows.append({'step':step,'sim_time_s':before,'phase':phase,'source':source,
                    'action':action.tolist(),'state_after':env.observation()['state'].tolist(),
                    'physics_qpos':env.data.qpos.tolist(),'physics_qvel':env.data.qvel.tolist(),
                    'inference_wall_s':inference_s,'outcome':info['outcome']})
                if terminated or truncated:
                    outcome=info['outcome'];reason=info['termination_reason'];break
            if capture is not None:
                spool=capture.finish(outcome if outcome in {'success','failure'} else 'aborted','branch_'+reason)
                # A successful downstream rollout is a candidate, not automatic admission.
                segments.append({'spool':str(spool),'frames':capture.count,'source':'scripted_expert',
                    'branch_outcome':outcome,'review_required':True,'training_eligible':False})
            path=output/f'{branch}-{repeat}.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            state_codec.dump(output/f'{branch}-{repeat}-final.json',{'environment':env.snapshot(),'policy':policy.snapshot()})
            result={'branch':branch,'repeat':repeat,'lineage':lineage,'split_group':lineage,
                'snapshot_sha256':digest,'initial_snapshot_sha256':initial,
                'outcome':outcome or 'censored','termination_reason':reason,'success':outcome=='success',
                'steps':len(rows),'sim_seconds':len(rows)*env.dt,'reset_step':reset_step,
                'waiting_sim_s':sum(r['phase']=='waiting' for r in rows)*env.dt,
                'correction_sim_s':sum(r['phase']=='correction' for r in rows)*env.dt,
                'matched_hold_sim_s':sum(r['phase']=='matched_hold' for r in rows)*env.dt,
                'operator_wall_seconds':None,'inference_wall_seconds':sum(r['inference_wall_s'] for r in rows),
                'trace':path.name}
            results.append(result)
            if repeat==0 and config.get('render_preview',False):
                from PIL import Image
                Image.fromarray(env.render()).save(output/(branch+'.png'))
            key=branch
            numeric=np.array([r['action']+r['physics_qpos']+r['physics_qvel'] for r in rows])
            if repeat==0:replay[key]=(numeric,outcome,reason)
            else:
                expected,prior_outcome,prior_reason=replay[key]
                same_shape=numeric.shape==expected.shape
                error=float(np.max(np.abs(numeric-expected))) if same_shape and numeric.size else 0. if same_shape else None
                result['replay_max_absolute_error']=error
                result['replay_verified']=bool(same_shape and error is not None and error<=1e-9 and outcome==prior_outcome and reason==prior_reason)
                if not result['replay_verified']:raise RuntimeError('Branch replay exceeded exact-state tolerance')
    first={r['branch']:r for r in results if r['repeat']==0}
    def difference(a,b):
        return None if 'censored' in {first[a]['outcome'],first[b]['outcome']} else int(first[a]['success'])-int(first[b]['success'])
    report={'status':'completed','protocol':'R5_protocol_adaptation_v1','domain':'sim','lineage':lineage,
        'split_group':lineage,'snapshot_sha256':digest,'model':env.manifest,'policy':policy.metadata,
        'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(state_codec.__file__))},
        'budget':config,'decision_context':decision_context,
        'actual_windows':{'delay_s':wait_steps*env.dt,'correction_s':help_steps*env.dt,'horizon_s':horizon_steps*env.dt},
        'outcomes':results,'correction_candidates':segments,
        'comparisons':{'correction_minus_continue_success':difference('correction','continue'),
            'correction_minus_hold_success':difference('correction','hold_only'),
            'hold_minus_continue_success':difference('hold_only','continue')},
        'reset_rule':'intervention arms reset policy queues at delay+correction boundary; continuation restores original queues',
        'inference_clock':'physics pauses during inference in every branch; latency reported, no operator timing claim',
        'wall_seconds':time.monotonic()-started,'training_started':False,'human_effect_verified':False,
        'replay_verified':all(r.get('replay_verified',True) for r in results),
        'limitations':['one independent lineage; repeats are not independent trials or probability estimates',
            'geometric correction uses simulator truth; not a human or a learned oracle',
            'censored horizon outcomes are not silently failures','model is not hardware calibrated']}
    (output/'result.json').write_text(json.dumps(report,indent=2))
    return report


def run(config,output):
    validate_config(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=make(config['profile']);policy=None
    try:
        env.reset(seed=config['seed'],options={'variation':config.get('variation','nominal')})
        if config.get('checkpoint'):
            if config.get('expected_model_hash')!=env.manifest['xml_sha256']:
                raise ValueError('Checkpoint training model and current model differ; explicit transfer protocol required')
            policy=FrozenPolicy(env,config['checkpoint'],output/'policy.log',config.get('device','cpu'))
        else:policy=DiagnosticHold(env)
        return experiment(env,policy,output,config)
    finally:
        if policy is not None:policy.close()
        env.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(run(json.loads(args.config.read_text()),args.output)))
