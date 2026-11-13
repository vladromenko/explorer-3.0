"""Provisional ROSMASTER model with the saved native six-channel servo contract."""
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
        result[5]=125-q[5]/.055*95
        return result

    def to_radians(self,servo):
        q=np.deg2rad(np.asarray(servo)-90)
        q[0]=np.deg2rad(90-servo[0])
        q[5]=(125-servo[5])/95*.055
        return q

    def gripper_is_open(self):
        return self.data.qpos[5]>.025
