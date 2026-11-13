"""Original Yahboom ROSMASTER geometry with the saved native six-channel servo contract."""
import numpy as np
from .environment import SO101PickPlace
from .models import rosmaster_arm_model


class RosmasterPickPlace(SO101PickPlace):
    fps=20
    profile='rosmaster_pick_place_sim'
    camera='front'
    camera_name='front_cam'
    model_builder=staticmethod(rosmaster_arm_model)
    home_native=np.array([90.,130.,0.,25.,90.,30.])

    def from_radians(self,q):
        result=np.rad2deg(np.asarray(q)).copy()+90
        result[0]=90-np.rad2deg(q[0])
        result[5]=30+q[5]/self.manifest['gripper_closed_rad']*95
        return result

    def to_radians(self,servo):
        q=np.deg2rad(np.asarray(servo)-90)
        q[0]=np.deg2rad(90-servo[0])
        q[5]=(servo[5]-30)/95*self.manifest['gripper_closed_rad']
        return q

    def gripper_is_open(self):
        return self.data.qpos[5]>.5*self.manifest['gripper_closed_rad']

    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        # Match the source URDF's five mimic constraints at initialization.
        self.data.qpos[6:11]=self.data.qpos[5]*np.array([1,1,-1,-1,1])
        import mujoco
        mujoco.mj_forward(self.model,self.data)
        return self.observation(),self.info()
