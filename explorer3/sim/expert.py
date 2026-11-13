"""Privileged geometric teacher, explicitly NOT a learned policy or a human."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class ScriptedSO101:
    def __init__(self, env):
        self.env = env
        self.scratch = mujoco.MjData(env.model)
        self.orientation = env.data.site('gripperframe').xmat.reshape(3,3).copy()
        self.cube = env.data.body('task_cube').xpos.copy()
        self.box = env.data.body('target_box').xpos.copy()
        self.phase = 'approach'

    def solve(self, xyz, gripper):
        m,d = self.env.model,self.scratch
        d.qpos[:] = self.env.data.qpos
        jacp,jacr = np.zeros((3,m.nv)),np.zeros((3,m.nv))
        for _ in range(80):
            mujoco.mj_forward(m,d)
            site = d.site('gripperframe')
            errp = xyz-site.xpos
            errr = Rotation.from_matrix(self.orientation @ site.xmat.reshape(3,3).T).as_rotvec()
            if np.linalg.norm(errp)<.0008 and np.linalg.norm(errr)<.02:
                break
            mujoco.mj_jacSite(m,d,jacp,jacr,m.site('gripperframe').id)
            jac = np.concatenate([jacp[:,:5],.08*jacr[:,:5]])
            err = np.concatenate([errp,.08*errr])
            delta = jac.T @ np.linalg.solve(jac @ jac.T+np.eye(6)*1e-5,err)
            d.qpos[:5] = np.clip(d.qpos[:5]+np.clip(delta,-.12,.12),self.env.ranges[:5,0],self.env.ranges[:5,1])
        q=d.qpos[:6].copy(); q[5]=gripper
        return self.env.from_radians(q)

    def action(self):
        t=self.env.elapsed
        if t<2:
            target=self.cube+[0,0,.09]; grip=.9; self.phase='approach'
        elif t<4:
            target=self.cube+[0,0,.002]; grip=.9; self.phase='descend'
        elif t<6:
            target=self.cube+[0,0,.002]; grip=-.12; self.phase='grasp'
        elif t<8:
            target=self.cube+[0,0,.13]; grip=-.12; self.phase='lift'
        elif t<11:
            target=self.box+[0,0,.14]; grip=-.12; self.phase='transfer'
        elif t<13:
            target=self.box+[0,0,.065]; grip=-.12; self.phase='lower'
        elif t<15:
            target=self.box+[0,0,.065]; grip=.9; self.phase='release'
        else:
            target=self.box+[0,0,.16]; grip=.9; self.phase='withdraw'
        return self.solve(target,grip)


class ScriptedRosmaster:
    """The existing console's IK, used only as a privileged simulation teacher."""
    def __init__(self,env):
        from .vendor_rosmaster.arm_model import ArmModel
        self.env=env;self.arm=ArmModel()
        self.cube=env.data.body('task_cube').xpos.copy()
        self.box=env.data.body('target_box').xpos.copy()
        self.phase='approach';self.targets={}
        self.command=env.observation()['state'].astype(float).copy()

    def action(self):
        t=self.env.elapsed
        if t<2:target=self.cube+[0,0,.10];grip=30;phase='approach'
        elif t<4:target=self.cube+[0,0,.004];grip=30;phase='descend'
        elif t<6:target=self.cube+[0,0,.004];grip=150;phase='grasp'
        elif t<8:target=self.cube+[0,0,.15];grip=150;phase='lift'
        elif t<11:target=self.box+[0,0,.16];grip=150;phase='transfer'
        elif t<13:target=self.box+[0,0,.075];grip=150;phase='lower'
        elif t<15:target=self.box+[0,0,.075];grip=30;phase='release'
        else:target=self.box+[0,0,.17];grip=30;phase='withdraw'
        self.phase=phase
        if phase not in self.targets:
            seed=self.env.observation()['state']
            solution,pitch=self.arm.grasp_ik(target,seed)
            if solution is None:
                raise RuntimeError(f'Existing ROSMASTER IK cannot reach {phase}')
            self.targets[phase]=np.r_[solution,grip]
        # The physical console also interpolates poses; a position jump would throw the object.
        maximum=np.array([50,50,50,50,50,90])*self.env.dt
        self.command+=np.clip(self.targets[phase]-self.command,-maximum,maximum)
        return self.command.copy()
