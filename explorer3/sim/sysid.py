"""Reuse Playground sysid with immutable Explorer physical trajectories.

UART write completion is a host timestamp proxy, NOT MCU receipt. Joint zeros
are provisional; results are diagnostic candidates and never auto-deployed.
"""
import json
from pathlib import Path
import numpy as np
from mujoco import sysid
from mujoco_playground.robots.so101.sim import scene
from mujoco_playground.tasks.sysid.so101 import data_loader
from mujoco_playground.tasks.sysid.so101.sysid import build_parameters
from explorer3 import catalog
from explorer3.paths import DATA
from explorer3.resources import ResourceLease


def trajectory(entry):
    path=Path(entry['dataset'])
    provenance=json.loads((path/'provenance.json').read_text())
    if provenance.get('synthetic') is not False or entry['profile']!='so101_pick_place':
        raise ValueError('Actual SO-101 recording required')
    rows=[json.loads(line) for line in (path/'research.jsonl').read_text().splitlines()]
    obs=np.array([r['observation_timestamp_ns'] for r in rows],dtype=np.int64)
    sent=np.array([r['action_send_end_ns'] for r in rows],dtype=np.int64)
    q=np.array([r['observation.state'] for r in rows],dtype=float)
    a=np.array([r['action'] for r in rows],dtype=float)
    for values in (q,a):
        values[:,:5]=np.deg2rad(values[:,:5])
        values[:,5]=-.174533+values[:,5]/100*(1.7453292+.174533)
    times=(obs-obs[0])*1e-9
    return data_loader.Trajectory(path=path,timestamps=times,ctrl_timestamps=(sent-obs[0])*1e-9,
        dt=float(np.mean(np.diff(times))),measured_qpos=q,commanded_ctrl=a,
        measured_qvel=np.gradient(q,times,axis=0),leader_timestamps_ns=obs,follower_timestamps_ns=sent)


def run(max_iters=3):
    output=DATA/'validation/corrective-a/sysid';output.mkdir(parents=True,exist_ok=True)
    episodes=sorted([e for e in catalog.rows('episodes') if e['profile']=='so101_pick_place'
                     and e['outcome']=='success' and not e['excluded']],key=lambda e:e['created'])
    if len(episodes)<3:raise ValueError('At least three existing physical trajectories required')
    chosen=episodes[:3]
    train=[trajectory(e) for e in chosen[:2]];held=[trajectory(chosen[2])]
    params=build_parameters()
    train_sequences=data_loader.to_model_sequences(train,scene.so101())
    test_sequences=data_loader.to_model_sequences(held,scene.so101())
    train_fn=sysid.build_residual_fn(models_sequences=[train_sequences])
    test_fn=sysid.build_residual_fn(models_sequences=[test_sequences])
    def rmse(fn,p):
        residuals,_,_=fn(p.as_vector(),p)
        values=np.concatenate([r.ravel() for r in residuals])
        return float(np.sqrt(np.mean(values**2)))
    baseline={'train_rad':rmse(train_fn,params),'held_trajectory_rad':rmse(test_fn,params)}
    (output/'plan.json').write_text(json.dumps({'fit_episodes':[e['id'] for e in chosen[:2]],
        'held_out_episode':chosen[2]['id'],'max_iterations':max_iters,'parameters':params.size,
        'timing':'host_uart_write_completion_proxy_not_mcu_receipt','calibration':'unverified_joint_zeros',
        'split_limitation':'held-out trajectory, not independent scene or operator'},indent=2))
    fitted,optimizer=sysid.optimize(params,train_fn,optimizer='mujoco',max_iters=max_iters)
    result={'implementation':'upstream Playground build_parameters/to_model_sequences + mujoco.sysid.optimize',
            'baseline':baseline,'fitted':{'train_rad':rmse(train_fn,fitted),'held_trajectory_rad':rmse(test_fn,fitted)},
            'parameters':{name:np.asarray(p.value).tolist() for name,p in fitted.items()},
            'optimizer_message':str(optimizer.get('message','bounded optimizer finished')),'automatic_application':False,'model_calibrated':False,
            'limitation':'Unknown physical joint-zero alignment and MCU receipt; diagnostic only. No parameter search on held-out trajectory.'}
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result


if __name__=='__main__':
    with ResourceLease('compute'):run()
