"""Read-only camera → real HTTP MJPEG acceptance, with event-loop latency measurement."""
import asyncio,json,threading,time
from types import SimpleNamespace
import cv2,httpx,numpy as np,uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from explorer3.hardware import CameraCapture,hardware_settings
from explorer3.preview import LivePreview
from explorer3.resources import ResourceLease
from explorer3.paths import DATA

async def main():
    cfg=hardware_settings()
    camera=CameraCapture('wrist',cfg['cameras']['wrist'],cfg)
    fake=SimpleNamespace(lock=threading.RLock(),control=SimpleNamespace(session='camera-readonly'),latest=None)
    preview=LivePreview(fake)
    app=FastAPI()
    @app.get('/preview')
    async def frames(request:Request):
        return StreamingResponse(preview.stream(request),media_type='multipart/x-mixed-replace; boundary=frame')
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8032,log_level='error',access_log=False))
    done=asyncio.Event()
    ticks=[]
    async def producer():
        while not done.is_set():
            frame=camera.latest_before(time.monotonic_ns())
            if frame:
                with fake.lock:
                    fake.latest={'images':{'wrist':frame}}
            ticks.append(time.monotonic())
            await asyncio.sleep(.02)
    with ResourceLease('cameras'):
        camera.start()
        serving=asyncio.create_task(server.serve())
        producing=asyncio.create_task(producer())
        try:
            while not server.started:
                await asyncio.sleep(.02)
            arrivals=[]
            pending=b''
            async with httpx.AsyncClient(timeout=10) as client:
                async with client.stream('GET','http://127.0.0.1:8032/preview') as response:
                    response.raise_for_status()
                    async for data in response.aiter_bytes():
                        pending+=data
                        parsed=True
                        while parsed:
                            split=pending.find(b'\r\n\r\n')
                            parsed=False
                            if split>=0:
                                headers=pending[:split].split(b'\r\n')
                                length=int(next(h.split(b':',1)[1] for h in headers if h.startswith(b'Content-Length:')))
                                end=split+4+length
                                if len(pending)>=end+2:
                                    image=cv2.imdecode(np.frombuffer(pending[split+4:end],dtype=np.uint8),cv2.IMREAD_COLOR)
                                    assert image.shape==(480,640,3)
                                    arrivals.append(time.monotonic())
                                    pending=pending[end+2:]
                                    parsed=True
                        if len(arrivals)>=60:
                            break
            intervals=np.diff(arrivals)
            result={'motor_ports_opened':False,'frames_decoded':len(arrivals),'received_fps':(len(arrivals)-1)/(arrivals[-1]-arrivals[0]),
                    'frame_gap_ms_p50_p95_max':(np.percentile(intervals,[50,95,100])*1000).tolist(),
                    'event_loop_max_gap_ms':max(np.diff(ticks))*1000,'camera':camera.info,
                    'browser_visual_result':'requires_operator_check','transport':'PASS'}
            (DATA/'validation/mjpeg-transport.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='camera'},indent=2))
        finally:
            done.set()
            await producing
            server.should_exit=True
            await serving
            camera.close()
asyncio.run(main())
