"""Checkpoint evaluation on local recordings. No robot imports or actuator access."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def evaluate(checkpoint, datasets, destination, max_samples=32, role='diagnostic'):
    checkpoint = Path(checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = 'cuda'
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}})
    reports = []
    for root in datasets:
        ds = LeRobotDataset('local/evaluation', root=root, video_backend='pyav')
        delta_timestamps = resolve_delta_timestamps(cfg, ds.meta)
        ds = LeRobotDataset('local/evaluation', root=root, video_backend='pyav',
                            delta_timestamps=delta_timestamps)
        errors, predictions, durations, losses = [], [], [], []
        for index in np.linspace(0, len(ds)-1, min(len(ds), max_samples), dtype=int):
            frame = ds[int(index)]
            batch = default_collate([frame])
            prepared = pre(batch)
            policy.reset()  # Each sampled observation is an independent replan, no stale chunk.
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                action = post(policy.select_action(prepared)).cpu().numpy()[0]
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            with torch.inference_mode():
                loss, _ = policy.forward(pre(default_collate([frame])))
            if not np.isfinite(action).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite policy output or validation loss')
            target = frame['action'].numpy()
            target = target[0] if target.ndim == 2 else target
            if action.shape != target.shape:
                raise ValueError('Checkpoint/action schema mismatch')
            predictions.append(action)
            errors.append(action - target)
            losses.append(float(loss.cpu()))
        errors = np.asarray(errors)
        reports.append({'dataset': str(root), 'samples': len(errors), 'channels': ds.features['action']['names'],
            'mae_by_channel': np.abs(errors).mean(axis=0).tolist(),
            'rmse_by_channel': np.sqrt((errors**2).mean(axis=0)).tolist(),
            'action_min': np.min(predictions, axis=0).tolist(), 'action_max': np.max(predictions, axis=0).tolist(),
            'official_forward_loss_mean': float(np.mean(losses)),
            'replan_latency_seconds_p50_p95_max': np.percentile(durations, [50,95,100]).tolist()})
    result = {'checkpoint': str(checkpoint), 'policy': cfg.type, 'role': role, 'datasets': reports,
        'normalization': 'checkpoint_train_processors_only', 'motor_commands_sent': False,
        'physical_skill_verified': False, 'limitation': 'Offline ошибки относительно показов не измеряют успешность задачи; sampled replan latency includes postprocessing, excludes sensors/network.'}
    Path(destination).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', default='diagnostic')
    parser.add_argument('--max-samples', type=int, default=32, choices=range(1,257))
    args = parser.parse_args()
    evaluate(args.checkpoint, args.dataset, args.output, args.max_samples, args.role)
