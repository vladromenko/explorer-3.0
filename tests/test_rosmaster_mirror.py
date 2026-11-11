import asyncio
import pytest
import websockets
from explorer3.rosmaster_mirror import packet, stream

class Reader:
    def sample(self):
        return {'positions': [1., -2., 3., 4., 5., 26.]}

def test_protocol_keeps_degrees_and_gripper_percent():
    p=packet(Reader().sample())
    assert list(p['leader'].values()) == [1., -2., 3., 4., 5., 26.]
    with pytest.raises(ValueError): packet({'positions':[float('nan')]*6})

def test_connection_loss_ends_stream_without_reconnect():
    async def scenario():
        connections=[];messages=[]
        async def receiver(ws):
            connections.append(ws)
            messages.append(await ws.recv())
            await ws.close(code=1011,reason='test link loss')
        async with websockets.serve(receiver,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            with pytest.raises(websockets.ConnectionClosed):
                await stream(f'ws://127.0.0.1:{port}',Reader(),30,duration=1)
        assert len(connections)==1 and len(messages)==1
    asyncio.run(scenario())
