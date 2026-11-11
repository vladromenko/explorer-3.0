"""Adapt CubeBot's so101_mirror_sender protocol to our pinned, read-only Leader.

Mapping remains in the copied CubeBot MirrorMapper on ROSMASTER. No controller,
Follower or camera is opened here. A broken connection ends this run; it never
reconnects or enables mirror mode. The operator enables mirror in Console 8090.
"""
import argparse
import asyncio
import json
import math
import time

import websockets

from .hardware import JOINTS
from .leader import LeaderReader


def packet(sample):
    values = sample['positions']
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError('Six finite Leader positions required')
    return {'leader': dict(zip(JOINTS, values)), 't': time.time()}


async def stream(uri, reader, hz=30., duration=None):
    if not 1 <= hz <= 30:
        raise ValueError('Leader rate must be 1..30 Hz')
    count = 0
    async with websockets.connect(uri, open_timeout=3, close_timeout=1, ping_interval=2, ping_timeout=2) as ws:
        started = time.monotonic()
        print('Leader stream connected; mirror mode is controlled by Console 8090', flush=True)
        while duration is None or time.monotonic() - started < duration:
            tick = time.monotonic()
            sample = await asyncio.to_thread(reader.sample)
            await asyncio.wait_for(ws.send(json.dumps(packet(sample), allow_nan=False)), timeout=.15)
            count += 1
            await asyncio.sleep(max(0., 1 / hz - (time.monotonic() - tick)))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.0.177')
    parser.add_argument('--port', default='/dev/so101-leader')
    parser.add_argument('--hz', type=float, default=30.)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    reader = LeaderReader(args.port)
    try:
        print(json.dumps(reader.start()), flush=True)
        asyncio.run(stream(f'ws://{args.host}:8090/mirror', reader, args.hz, args.seconds))
    finally:
        reader.close()
        print('Leader released; restart requires an explicit new run', flush=True)


if __name__ == '__main__':
    main()
