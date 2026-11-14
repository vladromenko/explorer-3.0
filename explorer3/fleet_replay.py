"""Build a bounded visual timeline from saved ROS messages, without publishing."""
import base64
import json
from pathlib import Path
import numpy as np
from agents.tank.sensors import decode_image
from .sensor_pages import previews


class ReplayIndex:
    def __init__(self,root):
        self.root=Path(root);self.latest={};self.frames=[];self.start=None;self.previous=None

    def observe(self,topic,msg,received_ns):
        parts=topic.split('/')
        if len(parts)!=5 or parts[1:3]!=['explorer3','robots']:return
        robot,key=parts[3:]
        if robot not in ('tank','rosmaster') or key not in ('rgb','depth','scan','scan_rear'):return
        info={'received_ns':received_ns,'source_timestamp_ns':msg.header.stamp.sec*10**9+msg.header.stamp.nanosec}
        if key in ('rgb','depth'):
            array=decode_image(msg)
            info.update(encoding=msg.encoding)
            if key=='depth':info['unit']='mm' if msg.encoding=='16UC1' else 'm'
        else:
            array=np.asarray(msg.ranges,dtype=np.float32)
            info.update(angle_min=msg.angle_min,angle_increment=msg.angle_increment,range_min=msg.range_min,range_max=msg.range_max)
        self.latest[robot,key]=(array,info)
        if self.start is None:self.start=received_ns
        if self.previous is None or received_ns-self.previous>=1_000_000_000:
            self.render(received_ns)

    def render(self,now):
        if len(self.frames)>=65:return
        frame={'after_s':round((now-self.start)/1e9,2),'received_ns':now,'views':{}}
        for robot in ('tank','rosmaster'):
            arrays={};streams={}
            for (owner,key),(array,info) in self.latest.items():
                age=max(now-info['received_ns'],now-info['source_timestamp_ns'])
                if owner==robot and now>=info['received_ns'] and 0<=age<=1_500_000_000:
                    arrays[key]=array;streams[key]=info
            images=previews({'arrays':arrays,'meta':{'streams':streams}})
            for key,encoded in images.items():
                name=f'preview-{len(self.frames):03d}-{robot}-{key}.jpg'
                (self.root/name).write_bytes(base64.b64decode(encoded.split(',',1)[1]))
                frame['views'][robot+'/'+key]={'file':name,'source_timestamp_ns':streams[key]['source_timestamp_ns'],
                    'received_ns':streams[key]['received_ns'],'age_s':round(max(now-streams[key]['received_ns'],now-streams[key]['source_timestamp_ns'])/1e9,3)}
        self.frames.append(frame);self.previous=now

    def finish(self):
        if self.latest:
            last=max(info['received_ns'] for _,info in self.latest.values())
            if last>self.previous:self.render(last)
        value={'version':1,'source':'official MCAP readback','display_only':True,
            'hardware_synchronized':False,'common_map_calibrated':False,'frames':self.frames}
        (self.root/'replay.json').write_text(json.dumps(value,ensure_ascii=False))
        return len(self.frames)
