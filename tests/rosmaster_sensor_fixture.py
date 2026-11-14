"""Synthetic sensor feed for recorder integration tests; no sockets or ROS."""
import io
import json
import time
import numpy as np
from agents.tank.sensors import SensorStore
from explorer3.sensors import SensorFeed,decode_packet


class FixtureFeed(SensorFeed):
    def _run(self):
        store=SensorStore('rosmaster')
        store.boot_id='synthetic-rosmaster-test'
        seq=0
        while not self.done.is_set():
            seq+=1
            store.put('rgb',np.full((48,64,3),seq%255,np.uint8),{'encoding':'rgb8'})
            store.put('depth',np.full((12,16),1000+seq,np.uint16),{'unit':'mm'})
            for key,value in [('scan',1.),('scan_rear',2.)]:
                store.put(key,np.full(360,value,np.float32),{'angle_min':0.,'angle_increment':np.pi/180,
                    'range_min':.05,'range_max':12.})
            store.put('imu',np.array([0,0,0,0,0,9.81],np.float32),{})
            meta,arrays=store.capture();buf=io.BytesIO()
            np.savez_compressed(buf,metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8),**arrays)
            packet=decode_packet(buf.getvalue())
            with self.history_lock:self.history.append(packet)
            self.latest=packet;self.done.wait(.2)


if __name__=='__main__':
    # Explicit fixture server for a separate browser-test process only.
    import os
    from pathlib import Path
    import uvicorn
    from fastapi import FastAPI,Request
    from fastapi.responses import Response
    cfg=json.loads((Path(os.environ['EXPLORER3_DATA'])/'configs/rosmaster-agent.json').read_text())
    feed=FixtureFeed(retain_history=True);feed.start();app=FastAPI()
    @app.get('/snapshot.npz')
    def snapshot(request:Request):
        if request.headers.get('X-Explorer-Token')!=cfg['token']:return Response(status_code=403)
        if feed.latest is None:return Response(status_code=503)
        return Response(feed.latest['payload'],media_type='application/octet-stream')
    try:uvicorn.run(app,host='127.0.0.1',port=8053,access_log=False)
    finally:feed.close()
