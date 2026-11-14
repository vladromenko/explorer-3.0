"""One policy runtime for every place a trained policy runs: MuJoCo evaluation, recorded-data
replay, shadow mode on a live robot, real trials and the swarm. The same code builds the batch
(state, camera images, language instruction), runs the official LeRobot pre/post-processors and
returns the action — so «it works in MuJoCo» and «it runs on the robot» test the same thing."""
import hashlib
import math
from pathlib import Path


def checkpoint_fingerprint(checkpoint):
    checkpoint = Path(checkpoint)
    return hashlib.sha256(b''.join(p.name.encode() + hashlib.sha256(p.read_bytes()).digest()
                                   for p in sorted(checkpoint.iterdir()) if p.is_file())).hexdigest()


class PolicyCore:
    TYPES = {'act', 'smolvla', 'diffusion'}

    def __init__(self, checkpoint, device='cuda', instruction=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        self.torch = torch
        self.checkpoint = Path(checkpoint)
        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        if cfg.type not in self.TYPES:
            raise ValueError('Unsupported LeRobot policy: ' + str(cfg.type))
        if cfg.type == 'smolvla' and getattr(cfg, 'rtc_config', None) is not None:
            raise ValueError('RTC state is not supported')
        cfg.device = device
        self.cfg, self.device, self.type = cfg, device, cfg.type
        self.policy = get_policy_class(cfg.type).from_pretrained(self.checkpoint, config=cfg).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(self.checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.instruction = instruction
        self.environment = cfg.input_features.get('observation.environment_state')
        self.image_keys = [k for k in cfg.input_features if k.startswith('observation.images.')]
        self.policy.reset()

    def reset(self):
        self.policy.reset()

    def needs_inference(self):
        queue = getattr(self.policy, '_action_queue', getattr(self.policy, '_queues', {}).get('action'))
        return queue is None or len(queue) == 0

    def batch(self, state, images, task, environment=None):
        """state: list; images: {camera name or full key: HxWx3 uint8 RGB}; task: text."""
        import numpy as np
        torch = self.torch
        b = {'observation.state': torch.tensor([list(state)], dtype=torch.float32), 'task': [task]}
        if self.environment is not None:
            if environment is None or len(environment) != self.environment.shape[0] or not np.isfinite(environment).all():
                raise ValueError('Policy requires valid environment observations')
            b['observation.environment_state'] = torch.tensor([list(environment)], dtype=torch.float32)
        for name, rgb in images.items():
            key = name if name.startswith('observation.images.') else 'observation.images.' + name
            b[key] = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().unsqueeze(0) / 255.
        missing = [k for k in self.image_keys if k not in b]
        if missing:
            raise ValueError('Policy camera missing: ' + ', '.join(missing))
        return b

    def act(self, state, images, task=None, environment=None):
        task = self.instruction or task
        if not task:
            raise ValueError('No task/instruction for the policy')
        with self.torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(self.batch(state, images, task, environment))))
        action = action.reshape(-1).cpu().tolist()
        if not all(math.isfinite(a) for a in action):
            raise ValueError('Nonfinite policy action')
        return action
