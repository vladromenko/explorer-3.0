"""Conservative measured response from archived sensors; never command odometry."""
import math
import numpy as np
from .sensors import stream_age


class ScanResponse:
    def __init__(self):self.previous=None

    def observe(self,packet):
        result={'yaw_rate_imu':None,'forward_displacement_lidar':None,
            'measured_response_kind':'imu_and_front_scan_pair_estimate_not_wheel_odometry',
            'lidar_response_valid':False,'lidar_response_reason':'Нужны два свежих скана'}
        arrays=packet['arrays'];streams=packet['meta']['streams']
        imu=arrays.get('imu')
        if imu is not None and len(imu)>=3 and np.isfinite(imu[:3]).all() and stream_age(packet,'imu')<=.5:
            result['yaw_rate_imu']=float(imu[2])
        scan=arrays.get('scan');meta=streams.get('scan')
        if scan is None or meta is None or stream_age(packet,'scan')>.5:
            return result
        current=(np.asarray(scan).copy(),dict(meta),packet['meta'].get('boot_id'))
        previous=self.previous;self.previous=current
        if previous is None:return result
        old,old_meta,boot=previous
        dt=(meta['received_ns']-old_meta['received_ns'])/1e9
        same_grid=all(meta.get(k)==old_meta.get(k) for k in ('angle_min','angle_increment'))
        if boot!=current[2] or old.shape!=scan.shape or not same_grid or not 0<dt<=.3:
            result['lidar_response_reason']='Повторный кадр, разрыв или другая геометрия скана';return result
        if result['yaw_rate_imu'] is None or abs(result['yaw_rate_imu'])*dt>.03:
            result['lidar_response_reason']='Поворот или отсутствие свежей IMU: линейная оценка исключена';return result
        angles=meta['angle_min']+np.arange(len(scan))*meta['angle_increment']
        wrapped=np.arctan2(np.sin(angles),np.cos(angles))
        valid=(np.abs(wrapped)<=math.radians(30))&np.isfinite(scan)&np.isfinite(old)
        valid&=(scan>=meta['range_min'])&(scan<=meta['range_max'])&(old>=old_meta['range_min'])&(old<=old_meta['range_max'])
        difference=(old[valid]-scan[valid])*np.cos(angles[valid])
        if len(difference)<8:return result
        displacement=float(np.median(difference));dispersion=float(np.median(np.abs(difference-displacement)))
        result.update(lidar_matching_points=len(difference),lidar_dispersion_m=dispersion,lidar_interval_s=dt)
        if dispersion>.03 or abs(displacement)/dt>.65:
            result['lidar_response_reason']='Сканы не согласуются с ограниченным прямым движением';return result
        result.update(forward_displacement_lidar=displacement,lidar_response_valid=True,
            lidar_response_reason='Смещение до поверхности впереди; движущиеся объекты не отделены')
        return result
