import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EXPLORER3_DATA', str(Path.home()/'.local/share/explorer3'))).resolve()

def prepare() -> None:
    for name in ('logs','datasets','checkpoints','cache','locks','jobs','validation'):
        (DATA/name).mkdir(parents=True, exist_ok=True)

def offline_environment() -> dict[str, str]:
    return {**os.environ, 'HF_HOME': str(DATA/'cache/huggingface'), 'TORCH_HOME': str(DATA/'cache/torch'),
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
            'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
            'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
