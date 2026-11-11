"""Prepare grouped splits then invoke the pinned official trainer, never a replacement loop."""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from .paths import DATA,ROOT,offline_environment
from .profiles import profile


def split_episodes(episodes: list[dict], seed=1000):
    groups={}
    for episode in episodes:
        groups.setdefault(episode['session']+'|'+episode['scene'],[]).append(episode)
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys)<3:
        raise ValueError('Для train/validation/test нужны не менее трёх независимых групп session/scene')
    hold=max(1,round(len(keys)*.1))
    selected={'test':keys[:hold],'validation':keys[hold:2*hold],'train':keys[2*hold:]}
    return {name:[e for k in subset for e in groups[k]] for name,subset in selected.items()}


def combine(episodes: list[dict], destination: Path, name: str):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    first=LeRobotDataset('local/source',root=episodes[0]['dataset'],video_backend='pyav')
    features={k:v for k,v in first.features.items() if k in {'observation.state','action'} or k.startswith('observation.images.')}
    ds=LeRobotDataset.create(repo_id='local/'+name,root=destination,fps=first.fps,features=features,
                            robot_type=first.meta.robot_type,use_videos=True,video_backend='pyav',image_writer_threads=2)
    for episode in episodes:
        src=LeRobotDataset('local/source',root=episode['dataset'],video_backend='pyav')
        if src.features != first.features or src.fps != first.fps:
            raise ValueError('Incompatible feature schema or fps')
        for frame in src:
            out={}
            for key in features:
                array=frame[key].numpy()
                out[key]=np.transpose(array,(1,2,0)) if key.startswith('observation.images.') else array
            out['task']=frame['task']
            ds.add_frame(out)
        ds.save_episode()
    ds.finalize()
    return ds


def train_command(dataset: Path, output: Path, policy='act',steps=50,batch=8,seed=1000):
    args=[sys.executable,'-m','lerobot.scripts.lerobot_train',f'--dataset.repo_id=local/{dataset.name}',
          f'--dataset.root={dataset}','--dataset.video_backend=pyav','--dataset.use_imagenet_stats=false',
          f'--policy.type={policy}','--policy.device=cuda','--policy.push_to_hub=false',
          '--wandb.enable=false',f'--output_dir={output}',f'--steps={steps}',f'--batch_size={batch}',
          '--num_workers=0','--persistent_workers=false',f'--seed={seed}','--env_eval_freq=0',
          '--log_freq=10',f'--save_freq={steps}']
    if policy=='smolvla':
        base=next((DATA/'cache/huggingface/hub/models--lerobot--smolvla_base/snapshots').iterdir())
        args.remove('--policy.type=smolvla')
        vlm=next((DATA/'cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots').iterdir())
        info=json.loads((dataset/'meta/info.json').read_text())
        features={}
        for key,value in info['features'].items():
            if key=='observation.state':
                features[key]={'type':'STATE','shape':value['shape']}
            elif key.startswith('observation.images.'):
                shape=value['shape']
                features[key]={'type':'VISUAL','shape':[shape[2],shape[0],shape[1]]}
        args.extend([f'--policy.path={base}',f'--policy.vlm_model_name={vlm}',
                     '--policy.input_features='+json.dumps(features),
                     '--policy.output_features='+json.dumps({'action':{'type':'ACTION','shape':info['features']['action']['shape']}})])
    return args


def run_job(path: Path):
    cfg=json.loads(path.read_text())
    profile(cfg['profile'])
    work=DATA/'jobs'/path.stem
    work.mkdir()
    splits={'train':cfg['episodes'],'validation':[],'test':[]} if cfg['smoke'] else split_episodes(cfg['episodes'],cfg['seed'])
    (work/'split.json').write_text(json.dumps(splits,indent=2))
    combine(splits['train'],work/'train',cfg['profile'])
    # Train statistics are computed exclusively from the materialized training episodes.
    steps=50 if cfg['smoke'] else 100000
    done=False
    for batch in ((1,) if cfg['policy']=='smolvla' else (8,4,2,1)):
        if not done:
            output=DATA/'checkpoints'/f'{path.stem}-b{batch}'
            command=train_command(work/'train',output,cfg['policy'],steps,batch,cfg['seed'])
            (work/f'command-b{batch}.json').write_text(json.dumps(command,indent=2))
            log_path=work/f'trainer-b{batch}.log'
            with log_path.open('w') as log:
                result=subprocess.run(command,env=offline_environment(),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:
                done=True
                checkpoint=output/'checkpoints/last/pretrained_model'
                subprocess.run([sys.executable,'-m','explorer3.reload_check',str(checkpoint),str(work/'train')],
                               env=offline_environment(),check=True)
                (work/'result.json').write_text(json.dumps({'checkpoint':str(checkpoint),'effective_batch':batch,'smoke':cfg['smoke'],'physical_skill_verified':False}))
            elif 'out of memory' not in log_path.read_text().lower() or batch==1:
                raise RuntimeError(f'Trainer failed: {log_path}')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    run_job(parser.parse_args().job)
