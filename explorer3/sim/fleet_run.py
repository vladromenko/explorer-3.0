"""Finite common-world inference job. CPU policies, shared physics, online monitor."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .fleet_execution import FleetExecution
from . import state_codec
from .fleet_capture import FleetCorrections


def atomic_json(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def run(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'progress.json',{'phase':'loading','training_started':False})
    runner=FleetExecution(config['specs'],output,seed=config['seed'],method=config['method'],
        delay_s=config['delay_s'],help_s=config['help_s'])
    started=time.monotonic();last_frame=0.;saved=False;replay={}
    captures=FleetCorrections(runner,config['seed']);runner.before_action=captures.observe;segments=None
    try:
        def progress(phase):
            status=runner.status();status.update(phase=phase,training_started=False)
            atomic_json(output/'progress.json',status)
            for name in (None,*runner.names):
                path=output/(('overview' if name is None else name)+'.jpg')
                temporary=path.with_suffix('.tmp');Image.fromarray(runner.world.render(name)).save(temporary,format='JPEG',quality=75);temporary.replace(path)
        while runner.world.data.time<config['horizon_s']-1e-10:
            captures.prepare_boundary();runner.advance();captures.end_completed()
            if time.monotonic()-last_frame>.5:progress('running');last_frame=time.monotonic()
            if not saved and runner.world.data.time>=min(.15,config['horizon_s']/2):
                snapshot=runner.snapshot();checksum=state_codec.dump(output/'snapshot.json',snapshot);saved=True
            if all(runner.world.views[n].done for n in runner.names):break
        segments=captures.finish();runner.before_action=None
        end=runner.snapshot();atomic_json(output/'commands.json',runner.rows)
        state_codec.dump(output/'final-state.json',end)
        if saved:
            progress('checking_restore')
            runner.restore(state_codec.load(output/'snapshot.json'))
            for _ in range(config['replay_steps']):runner.advance()
            expected=runner.world.data.qpos.copy();velocity=runner.world.data.qvel.copy();events=runner.events.copy()
            counts=runner.counts.copy();policy_states={n:p.snapshot() for n,p in runner.policies.items()}
            runner.restore(state_codec.load(output/'snapshot.json'))
            for _ in range(config['replay_steps']):runner.advance()
            np.testing.assert_allclose(runner.world.data.qpos,expected,rtol=0,atol=1e-9)
            np.testing.assert_allclose(runner.world.data.qvel,velocity,rtol=0,atol=1e-8)
            assert events==runner.events and counts==runner.counts
            # Full policy RNG and queued actions must match, not only the body pose.
            a=state_codec.dump(output/'expected-policy.json',policy_states)
            b=state_codec.dump(output/'replayed-policy.json',{n:p.snapshot() for n,p in runner.policies.items()})
            assert a==b,'Policy state diverged after restore'
            replay={'verified':True,'snapshot_sha256':checksum,'steps':config['replay_steps'],
                'max_qpos_error':float(np.max(np.abs(runner.world.data.qpos-expected))),
                'policy_state_sha256':a}
            runner.restore(end)
        progress('completed')
        result={**runner.status(),'status':'completed','domain':'sim','config':config,'replay':replay,
            'events':runner.events,'correction_segments':segments,'lineage':captures.lineage,'effort':runner.metrics(),
            'supervision_sim_seconds':{n:runner.counts[n]['scripted_expert']*runner.world.views[n].env.dt for n in runner.names},
            'wall_seconds':time.monotonic()-started,'training_started':False,
            'physical_robot_access':False,'physical_skill_verified':False,'human_effect_verified':False,
            'calibrated_gain':False,'command_trace':str(output/'commands.json'),
            'shared_model':runner.world.manifest,
            'policy_model_contracts':{n:runner.world.agent_manifest(n) for n in runner.names},
            'implementation_sha256':hashlib.sha256(Path(__file__).with_name('fleet_execution.py').read_bytes()).hexdigest(),
            'limits':['Physics pauses during inference; wall time is not operator effort',
                'Explicit numerical fixtures are not calibrated learned risk',
                'Old smoke weights in software probe do not demonstrate task skill',
                'No human correction or causal-gain claim','Shared nominal solver is a distinct transfer condition']}
        atomic_json(output/'result.json',result);return result
    finally:
        if segments is None:captures.finish(aborted=True)
        runner.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    result=run(json.loads(args.config.read_text()),args.output)
    print(json.dumps({'status':result['status'],'replay':result['replay'],'training_started':False}))
