"""Measured resident-policy load on recorded observations; no hardware or uncertainty claim."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .paths import DATA
from .trials import prepare, PolicyActions


def run(source_job, destination, copies=3, cycles=10):
    if copies not in {1,3} or not 1<=cycles<=30:raise ValueError('Unsupported benchmark size')
    plan=prepare('policy',source_job)
    ds=LeRobotDataset('local/load-probe',root=DATA/'jobs'/source_job/'train',video_backend='pyav')
    frame=ds[0]
    images={key.split('observation.images.',1)[1]:{'image':(value.permute(1,2,0).numpy()*255).round().astype('uint8')}
            for key,value in frame.items() if key.startswith('observation.images.')}
    observation={'observation.state':frame['observation.state'].tolist(),'images':images}
    policies=[PolicyActions(plan) for _ in range(copies)]
    for policy in policies:
        policy.next(observation);policy.reset()
    torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(cycles):
        torch.cuda.synchronize();start=time.perf_counter()
        for policy in policies:
            policy.reset()  # Force full replanning; no cached-action timing masquerading as inference.
            policy.next(observation)
        torch.cuda.synchronize();times.append(time.perf_counter()-start)
    result={'copies':copies,'cycles':cycles,'checkpoint':plan['checkpoint'],
            'same_weights_in_all_copies':True,'independent_ensemble':False,
            'combined_replan_seconds_p50_p95_max':np.percentile(times,[50,95,100]).tolist(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'all_replans_fit_50ms':max(times)<=.05,'all_replans_fit_33ms':max(times)<=1/30,
            'scope':'Sequential GPU scheduling with all models resident; real recorded SO-101 input reused',
            'excludes':['sensor latency','network','actuator execution','three different trained policies'],
            'motor_commands_sent':False,'physical_skill_verified':False,'simultaneous_robot_acceptance':False}
    Path(destination).write_text(json.dumps(result,indent=2))
    return result
