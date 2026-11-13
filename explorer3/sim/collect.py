"""Real executions of the geometric expert, with immutable failures included."""
import argparse
import json
import subprocess
from explorer3.paths import ROOT,DATA,offline_environment
from .capture import SimCapture
from .environment import SO101PickPlace
from .factory import make,EXPERTS


def collect(seeds,profile="so101_pick_place_sim"):
    env=make(profile)
    results=[]
    try:
        for seed in seeds:
            env.reset(seed=seed)
            expert=EXPERTS[profile](env)
            capture=SimCapture(env,seed,'scripted_expert')
            finished=False
            while not finished:
                action=expert.action()
                capture.add(env,action)
                _,_,terminated,truncated,info=env.step(action)
                finished=terminated or truncated
            root=capture.finish(info['outcome'],info['termination_reason'])
            subprocess.run([str(ROOT/'.venv/bin/python'),'-m','explorer3.sim.import_episode',str(root)],
                           check=True,env=offline_environment())
            results.append({'seed':seed,'outcome':info['outcome'],'spool':str(root)})
            print(json.dumps(results[-1]),flush=True)
    finally:
        env.close()
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=1);parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--profile',default='so101_pick_place_sim')
    args=parser.parse_args();collect(range(args.start,args.start+args.count),args.profile)
