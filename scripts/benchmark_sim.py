"""Measured sequential vector sizes, native physics and Homebrain-rendered camera."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import psutil
from explorer3.sim.factory import make


def run(profile,output):
    reports=[]
    for count in (1,2,4):
        started=time.monotonic();envs=[make(profile) for _ in range(count)]
        cold=time.monotonic()-started
        timings=[];frames=[]
        try:
            for index,env in enumerate(envs):env.reset(seed=500+index)
            start=time.monotonic()
            for step in range(120):
                for env in envs:
                    before=time.monotonic();env.step(env.observation()['state'][:env.action_space.shape[0]]);timings.append(time.monotonic()-before)
            elapsed=time.monotonic()-start
            before=time.monotonic();envs[0].render();render_cold=time.monotonic()-before
            for _ in range(20):
                before=time.monotonic();envs[0].render();frames.append(time.monotonic()-before)
            reports.append({'environments':count,'vectorization':'sequential native Gym environments',
                'construction_seconds':cold,'physics_steps_per_second':count*120/elapsed,
                'step_seconds_p50_p95_max':np.percentile(timings,[50,95,100]).tolist(),
                'camera_cold_seconds':render_cold,'camera_seconds_p50_p95_max':np.percentile(frames,[50,95,100]).tolist(),
                'rss_bytes':psutil.Process().memory_info().rss,'available_memory_bytes':psutil.virtual_memory().available})
        finally:
            for env in envs:env.close()
    Path(output).write_text(json.dumps({'profile':profile,'measurements':reports},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('profile');parser.add_argument('output');args=parser.parse_args();run(args.profile,args.output)
