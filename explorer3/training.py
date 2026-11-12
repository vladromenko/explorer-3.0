"""Prepare grouped splits then invoke the pinned official trainer, never a replacement loop."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .paths import DATA,ROOT,offline_environment
from .profiles import profile, TRAINING
from .datasets import grouped_split, load_plan


split_episodes = grouped_split

def combine(episodes: list[dict], destination: Path, name: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets
    sources=[LeRobotDataset('local/'+episode['id'],root=episode['dataset'],video_backend='pyav')
             for episode in episodes]
    # Upstream aggregation preserves the recorded MP4 files without a lossy re-encode.
    # Only the selected training sources contribute to the merged normalization statistics.
    return merge_datasets(sources,output_repo_id='local/'+name,output_dir=destination,
                          concatenate_videos=False,concatenate_data=False)


def train_command(dataset: Path, output: Path, policy='act',steps=50,batch=8,seed=1000):
    args=[sys.executable,'-m','lerobot.scripts.lerobot_train',f'--dataset.repo_id=local/{dataset.name}',
          f'--dataset.root={dataset}','--dataset.video_backend=pyav','--dataset.use_imagenet_stats=false',
          f'--policy.type={policy}','--policy.device=cuda','--policy.push_to_hub=false',
          '--wandb.enable=false',f'--output_dir={output}',f'--steps={steps}',f'--batch_size={batch}',
          '--num_workers=0','--persistent_workers=false',f'--seed={seed}','--env_eval_freq=0',
          '--log_freq=10',f'--save_freq={min(steps,5000)}']
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


def evaluation_command(checkpoint, roots, destination, role):
    return [sys.executable, '-m', 'explorer3.evaluation', '--checkpoint', str(checkpoint),
            '--output', str(destination), '--role', role] + [v for root in roots for v in ('--dataset', str(root))]


def run_job(path: Path):
    cfg = json.loads(path.read_text())
    work = DATA/'jobs'/path.stem
    work.mkdir(exist_ok=True)
    if cfg.get('kind') == 'benchmark':
        from .benchmark import run
        run(cfg['source_job'],work/'result.json')
        return
    if cfg.get('kind') == 'research':
        from .research import analyze
        analyze(work)
        return
    if cfg.get('kind') == 'evaluate':
        source = DATA/'jobs'/cfg['source_job']
        split = json.loads((source/'split.json').read_text())
        partition=cfg.get('partition','validation')
        if partition=='test':
            if not split['test']:raise ValueError('Smoke не содержит независимого test')
            import hashlib
            checksum=hashlib.sha256((Path(cfg['checkpoint'])/'model.safetensors').read_bytes()).hexdigest()
            choice=source/'frozen-test-checkpoint.json'
            if choice.exists() and json.loads(choice.read_text())['sha256']!=checksum:
                raise ValueError('Test уже открыт для другого checkpoint; нельзя подбирать по test')
            choice.write_text(json.dumps({'checkpoint':cfg['checkpoint'],'sha256':checksum}))
        selected = split[partition] or split['train']
        role = 'held_out_'+partition if split[partition] else 'training_data_diagnostic_not_generalization'
        subprocess.run(evaluation_command(cfg['checkpoint'], [e['dataset'] for e in selected], work/'evaluation.json', role),
                       env=offline_environment(), check=True)
        (work/'result.json').write_text(json.dumps({'checkpoint': cfg['checkpoint'], 'evaluation': str(work/'evaluation.json'),
                                                   'physical_skill_verified': False}))
        return
    profile(cfg['profile'])
    if cfg.get('plan_id'):
        plan = load_plan(cfg['plan_id'])
        splits = plan['splits']
    else:
        splits = json.loads((work/'split.json').read_text()) if (work/'split.json').exists() else (
            {'train': cfg['episodes'], 'validation': [], 'test': []} if cfg['smoke'] else split_episodes(cfg['episodes'], cfg['seed']))
    split_path = work/'split.json'
    if split_path.exists() and json.loads(split_path.read_text()) != splits:
        raise ValueError('Frozen split differs from previous run')
    split_path.write_text(json.dumps(splits, indent=2))
    if not (work/'train').exists():
        combine(splits['train'], work/'train', cfg['profile'])
    steps = TRAINING[cfg['policy']].get('smoke_steps', 50) if cfg['smoke'] else TRAINING[cfg['policy']].get('steps', 100000)
    resume = cfg.get('resume_checkpoint')
    if resume:
        saved = json.loads((Path(resume)/'train_config.json').read_text())
        batches = (saved['batch_size'],)
    else:
        batches = (1,) if cfg['policy']=='smolvla' else (8,4,2,1)
    done = False
    for batch in batches:
        if not done:
            output = DATA/'checkpoints'/f'{path.stem}-b{batch}'
            command = ([sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                        f'--config_path={resume}/train_config.json', '--resume=true'] if resume else
                       train_command(work/'train', output, cfg['policy'], steps, batch, cfg['seed']))
            (work/f'command-b{batch}.json').write_text(json.dumps(command, indent=2))
            log_path = work/f'trainer-b{batch}.log'
            with log_path.open('a') as log:
                result = subprocess.run(command, env=offline_environment(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                done = True
                checkpoint = output/'checkpoints/last/pretrained_model'
                subprocess.run([sys.executable, '-m', 'explorer3.reload_check', str(checkpoint), str(work/'train')],
                               env=offline_environment(), check=True)
                if splits['validation']:
                    subprocess.run(evaluation_command(checkpoint, [e['dataset'] for e in splits['validation']],
                                   work/'evaluation.json', 'held_out_validation'), env=offline_environment(), check=True)
                (work/'result.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'effective_batch': batch,
                    'seed': cfg['seed'], 'smoke': cfg['smoke'], 'steps': steps, 'plan_id': cfg.get('plan_id'),
                    'physical_skill_verified': False, 'test_used': False}))
            elif resume or 'out of memory' not in log_path.read_text().lower() or batch == 1:
                raise RuntimeError(f'Trainer failed: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    run_job(parser.parse_args().job)
