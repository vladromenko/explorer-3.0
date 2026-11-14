"""Synthetic timestamps/images only: managed four-profile recording, no robot/model."""
import io,json,sys,time
import numpy as np
from explorer3 import catalog,collection,profiles
from explorer3.recording import EpisodeWriter
from explorer3.paths import prepare
from explorer3.sensors import decode_packet,environment_features

mode=sys.argv[2] if len(sys.argv)>2 else 'normal'
name=sys.argv[1];task=name.removesuffix('_rgbd_lidar')
prepare();catalog.initialize()
cfg=profiles.PROFILES[name];cfg['domain']='sim'
series=collection.create_series(task,name,'fixture','Synthetic layout')
writer=EpisodeWriter(name,series['id'],series['layout'],domain='fixture')
base=1_000_000_000;step=int(1e9/cfg['fps'])
for i in range(int(cfg['fps']*3.5)):
    now=base+i*step+(400_000_000 if mode=='gap' and i>=20 else 0)
    sample={'observation_timestamp_ns':now,'observation.state':[0.]*len(cfg.get('state_channels',cfg['channels'])),
        'action':[0.]*len(cfg['channels']),'action_send_start_ns':now+1,'action_send_end_ns':now+2,
        'action_acknowledged':False,'command_source':'human','battery_voltage':7.8 if cfg['robot']=='tank' else 11.7,
        'images':{cfg['camera']:{'image':np.full(cfg.get('image_shape',[480,640,3]),i%255,np.uint8),
                               'timestamp_ns':now-1000,'sequence':i}}}
    if cfg.get('environment_channels'):
        arrays={'depth':np.full((12,16),1000,np.uint16),'scan':np.ones(360,np.float32),'imu':np.array([0,0,0,0,0,9.81],np.float32)}
        if cfg['robot']=='rosmaster':arrays['scan_rear']=np.ones(360,np.float32)
        streams={key:{'received_ns':now-1000,'source_timestamp_ns':now-1000,'age_s':0.,'frame_count':i+1} for key in arrays}
        streams['depth']['unit']='mm'
        for key in ('scan','scan_rear'):
            if key in streams:streams[key].update(angle_min=0.,angle_increment=np.pi/180,range_min=.05,range_max=12.)
        meta={'version':1,'robot':cfg['robot'],'boot_id':'fixture','monotonic_ns':now-1000,'streams':streams}
        buf=io.BytesIO();np.savez_compressed(buf,metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8),**arrays)
        packet=decode_packet(buf.getvalue(),received_ns=now-1000)
        sample.update(_sensor_packet=packet,**{'observation.environment_state':environment_features(packet,now).tolist()})
    writer.add(sample)
writer.finish('success',{'fixture':True})
end=time.monotonic()+60
while collection.summary(series['id'])['attempts'][0]['phase']!='saved' and time.monotonic()<end:time.sleep(.05)
report=collection.summary(series['id']);quality=json.loads(report['attempts'][0]['quality'])
assert quality['status']==('warning' if mode=='gap' else 'pass'),quality
assert (writer.root/'research.jsonl').exists()
assert list((writer.root/'videos').rglob('*.mp4'))
assert catalog.rows('episodes')[0]['excluded']==1,'Synthetic data must stay excluded'
print(json.dumps({'profile':name,'frames':writer.frames,'quality':quality['status'],'groups':report['independent_groups']}))
