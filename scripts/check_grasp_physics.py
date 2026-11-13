"""Finite mechanical acceptance, no trainer, no policy weights, no hardware."""
import argparse
import json
from pathlib import Path
import cv2
import mujoco
from explorer3.sim.factory import make,EXPERTS
from explorer3.sim.contact_audit import ContactAudit


def check(profile,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=make(profile);results=[];renderer=None
    try:
        renderer=mujoco.Renderer(env.model,height=480,width=640)
        for seed in (1,2,3):
            env.reset(seed=seed);expert=EXPERTS[profile](env)
            audit=ContactAudit(env.model);env.physics_observer=audit
            photographed=set()
            for _ in range(round(env.fps*env.deadline_s)+1):
                _,_,terminated,truncated,info=env.step(expert.action())
                if seed==1 and expert.phase in {'lift','transfer','release'} and expert.phase not in photographed:
                    # Mid-phase avoids showing only the transition's starting pose.
                    threshold={'lift':7.5,'transfer':10.,'release':14.5}[expert.phase]
                    if env.elapsed>=threshold:
                        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                        camera.lookat[:]=env.data.body('task_cube').xpos
                        camera.distance=.32;camera.azimuth=135;camera.elevation=-20
                        renderer.update_scene(env.data,camera=camera)
                        cv2.imwrite(str(output/f'{profile}-{expert.phase}.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
                        photographed.add(expert.phase)
                if terminated or truncated:break
            result={'seed':seed,'outcome':info['outcome'],'reason':info['termination_reason'],**audit.report()}
            result['pass']=bool(info['success'] and info['released'] and info['inside_box'] and audit.lifted
                and audit.minimum_distance_m>-.0005 and audit.maximum_mimic_error_rad<.001
                and audit.minimum_carry_height_m>.045)
            results.append(result);print(json.dumps(result),flush=True)
        report={'profile':profile,'model':env.manifest,'mujoco_version':mujoco.__version__,
                'controller':'scripted geometric trajectory; no learning','sampling':'every physics substep',
                'pass':all(row['pass'] for row in results),'results':results}
        (output/f'{profile}.json').write_text(json.dumps(report,indent=2))
        if not report['pass']:raise AssertionError('Contact acceptance failed; see report')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    check(args.profile,args.output)
