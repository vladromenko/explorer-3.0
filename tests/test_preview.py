import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from explorer3.preview import LivePreview
from explorer3.control import Control

class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_consumer_receives_latest_not_a_backlog(self):
        frame=np.zeros((480,640,3),dtype=np.uint8)
        session=SimpleNamespace(lock=threading.RLock(),control=SimpleNamespace(session='one'),
            latest={'images':{'wrist':{'sequence':1,'image':frame}}})
        request=SimpleNamespace(is_disconnected=lambda:asyncio.sleep(0,result=False))
        preview=LivePreview(session)
        stream=preview.stream(request)
        first=await anext(stream)
        self.assertIn(b'Content-Type: image/jpeg',first)
        for sequence in range(2,31):
            session.latest={'images':{'wrist':{'sequence':sequence,'image':frame}}}
        await anext(stream)
        self.assertEqual(preview.key,('one',30))
        self.assertEqual(preview.encoded,2)
        await stream.aclose()
        self.assertEqual(preview.clients,0)

    async def test_encoding_does_not_block_heartbeat_event_loop(self):
        frame=np.zeros((480,640,3),dtype=np.uint8)
        session=SimpleNamespace(lock=threading.RLock(),control=SimpleNamespace(session='one'),
            latest={'images':{'wrist':{'sequence':1,'image':frame}}})
        preview=LivePreview(session)
        def delayed_encode(rgb):
            import time
            time.sleep(.15)
            return b'jpeg'
        ticks=[]
        async def heartbeat():
            for _ in range(6):
                ticks.append(asyncio.get_running_loop().time())
                await asyncio.sleep(.025)
        with patch.object(preview,'encode',side_effect=delayed_encode):
            await asyncio.gather(preview.latest(),heartbeat())
        self.assertLess(max(b-a for a,b in zip(ticks,ticks[1:])),.1)

    async def test_cleanup_preserves_actual_stop_reason(self):
        with patch('explorer3.control.catalog.event'):
            c=Control()
            with patch.object(c,'authorized',return_value=True):
                c.acquire('browser')
            c.stop('Браузер: вкладка скрыта')
            c.stop('Физическая сессия закрывается')
            self.assertEqual(c.last_stop_reason,'Браузер: вкладка скрыта')
            c.heartbeat('browser')
            self.assertTrue(c.expired())

    async def test_shutdown_closes_idle_http_stream(self):
        stop=threading.Event()
        session=SimpleNamespace(lock=threading.RLock(),latest=None)
        request=SimpleNamespace(is_disconnected=lambda:asyncio.sleep(0,result=False))
        preview=LivePreview(session,stop_requested=stop.is_set)
        stream=preview.stream(request)
        pending=asyncio.create_task(anext(stream))
        await asyncio.sleep(.02)
        self.assertEqual(preview.clients,1)
        stop.set()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(pending,timeout=.3)
        self.assertEqual(preview.clients,0)
