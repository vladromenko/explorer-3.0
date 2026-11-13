"""Initial framing through Studio's official plugin API; rendering stays upstream."""
from mujoco.studio import plugin,messages


class TaskCamera(plugin.ViewerPlugin):
    @messages.handler
    def frame_task(self,event:messages.PostModelEvent):
        mobile=event.model.nq==3
        self.viewer.camera.lookat[:]=[.5,0,.1] if mobile else [.15,0,.15]
        self.viewer.camera.distance=2.1 if mobile else .75
        self.viewer.camera.azimuth=135
        self.viewer.camera.elevation=-25
