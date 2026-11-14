"""Physics/vision acceptance in the actual shared MuJoCo, no training/hardware."""
import hashlib
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from explorer3.paths import DATA
from explorer3.sim.room import PRESETS,ASSETS
from explorer3.sim.cooperation import CooperationWorld

out=DATA/'validation/cognitive/rooms';out.mkdir(parents=True,exist_ok=True)
lock=json.loads((ASSETS/'manifest.json').read_text())
for name,record in lock['files'].items():assert hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()==record['sha256']
reports=[]
for preset in PRESETS:
    instance=CooperationWorld(out/preset,17,preset);world=instance.world
    try:
        initial=instance.snapshot()
        assert not initial['recipient']['measurement']['target_visible']
        try:instance.dock();raise AssertionError('Dock started before donor evidence')
        except ValueError as exc:assert 'blocked' in str(exc)
        tracked=['prop_bottle','prop_can','prop_lemon','shared_ball']
        before={name:float(world.data.xpos[world.model.body(name).id,2]) for name in tracked}
        helper=instance.observe(scan=True);assert helper['helper_outcome']=='success' and helper['recipient_held']
        physics={}
        for name in tracked:
            bid=world.model.body(name).id
            jid=world.model.body_jntadr[bid];dof=world.model.jnt_dofadr[jid]
            contacts=[]
            for index in range(world.data.ncon):
                contact=world.data.contact[index]
                if bid in (world.model.geom_bodyid[contact.geom1],world.model.geom_bodyid[contact.geom2]):
                    force=np.zeros(6);mujoco.mj_contactForce(world.model,world.data,index,force)
                    contacts.append({'penetration_m':max(0.,-float(contact.dist)),'normal_force_n':float(force[0])})
            speed=float(np.linalg.norm(world.data.qvel[dof:dof+6]));height=float(world.data.xpos[bid,2])
            assert np.isfinite(world.data.qpos).all() and np.isfinite(world.data.qvel).all()
            assert contacts and sum(x['normal_force_n'] for x in contacts)>0,(preset,name,contacts)
            assert max(x['penetration_m'] for x in contacts)<.002,(preset,name,contacts)
            assert speed<.02,(preset,name,speed)
            assert before[name]-height>.003,(preset,name,before[name],height)
            physics[name]={'initial_z':before[name],'rest_z':height,'speed':speed,'contacts':contacts}
        outcome=instance.dock();assert outcome['task']=='success',outcome
        saved=world.snapshot();world.step();world.restore(saved)
        assert np.array_equal(saved['physics'],world.snapshot()['physics'])
        cv2.imwrite(str(out/preset/'overview.png'),cv2.cvtColor(world.render(),cv2.COLOR_RGB2BGR))
        reports.append({'preset':preset,'world_sha256':world.manifest['xml_sha256'],'physics':physics,
            'assistance':outcome,'exact_snapshot_restore':True,'target_not_visible_initially':True})
    finally:instance.close()

# An actual robot-object collision in the SAME MjData, with commanded drive
# forces. No teleporting the object during the measured interaction.
instance=CooperationWorld(out/'contact-probe',17,'workshop');w=instance.world
try:
    tank=w.views['tank'];w.data.qpos[tank.q]=[0.,0.,0.]
    joint=w.model.joint('shared_ball_free');adr=int(joint.qposadr[0])
    w.data.qpos[adr:adr+3]=[.27,1.1,.046]
    mujoco.mj_forward(w.model,w.data)
    for view in w.views.values():view.read(forward=True)
    start=w.data.qpos[adr:adr+3].copy();initial_state=w.snapshot();observed_contact=False;peak_force=0.;penetration=0.
    for _ in range(40):
        w.step({'tank':[.12,0.]})
        for index in range(w.data.ncon):
            c=w.data.contact[index];bodies=[w.model.body(int(w.model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
            if 'shared_ball' in bodies and any(name and name.startswith('tank/') for name in bodies):
                observed_contact=True;force=np.zeros(6);mujoco.mj_contactForce(w.model,w.data,index,force);peak_force=max(peak_force,float(force[0]))
                penetration=max(penetration,max(0.,-float(c.dist)))
    w.stop();end=w.data.qpos[adr:adr+3].copy()
    displacement=float(np.linalg.norm(end[:2]-start[:2]))
    obstructed_x=float(w.data.qpos[tank.q[0]])
    w.restore(initial_state);w.data.qpos[adr+1]+=.5;mujoco.mj_forward(w.model,w.data)
    for view in w.views.values():view.read(forward=True)
    for _ in range(40):w.step({'tank':[.12,0.]})
    unobstructed_x=float(w.data.qpos[tank.q[0]])
    assert observed_contact and peak_force>0 and penetration<.002,(observed_contact,peak_force,penetration)
    assert unobstructed_x-obstructed_x>.02,(obstructed_x,unobstructed_x)
    contact_report={'observed_robot_object_contact':observed_contact,'peak_normal_force_n':peak_force,
        'object_displacement_m':displacement,'contact_penetration_m':penetration,
        'obstructed_robot_x':obstructed_x,'unobstructed_robot_x':unobstructed_x,
        'counterfactual':'same snapshot and command, ball placed outside path before second branch; privileged diagnostic only',
        'single_mjdata':True,'drive_source':'bounded velocity command through existing force-based Tank model',
        'limitation':'planar chassis surrogate; not calibrated tracks or wheel-ground dynamics'}
    cv2.imwrite(str(out/'contact-probe'/'overview.png'),cv2.cvtColor(w.render(),cv2.COLOR_RGB2BGR))
finally:instance.close()
report={'passed':True,'rooms':reports,'robot_object_interaction':contact_report,'optimizer_steps':0,'physical_commands':0}
(out/'physics-acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
