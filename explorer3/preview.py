"""Read-only MJPEG preview with per-client backpressure, separate from control events."""
import asyncio
import cv2

class LivePreview:
    def __init__(self,session,stop_requested=lambda:False):
        self.session=session
        self.stop_requested=stop_requested
        self.key=None
        self.jpeg=None
        self.lock=asyncio.Lock()
        self.encoded=0
        self.clients=0

    @staticmethod
    def encode(rgb):
        ok,data=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:
            raise RuntimeError('Preview JPEG encoding failed')
        return data.tobytes()

    async def latest(self):
        with self.session.lock:
            sample=self.session.latest
        if sample is None:
            return None,None
        key=(self.session.control.session,sample['images']['wrist']['sequence'])
        async with self.lock:
            if key!=self.key:
                self.jpeg=await asyncio.to_thread(self.encode,sample['images']['wrist']['image'])
                self.key=key
                self.encoded+=1
            return self.key,self.jpeg

    async def stream(self,request):
        self.clients+=1
        previous=None
        try:
            while not self.stop_requested() and not await request.is_disconnected():
                key,jpeg=await self.latest()
                if jpeg is not None and key!=previous:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
                    previous=key
                await asyncio.sleep(1/15)
        finally:
            self.clients-=1
