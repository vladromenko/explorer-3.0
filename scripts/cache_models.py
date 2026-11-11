"""Copy only public upstream caches (never old robot checkpoints)."""
from pathlib import Path
import hashlib,json,shutil
from explorer3.paths import DATA
source=Path.home()/'.cache/huggingface/hub'
destination=DATA/'cache/huggingface/hub'
destination.mkdir(parents=True,exist_ok=True)
manifest=[]
for model in ('models--lerobot--smolvla_base','models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct'):
    src=source/model
    dst=destination/model
    shutil.copytree(src,dst,symlinks=False,dirs_exist_ok=True)
    for file in sorted(dst.rglob('*')):
        if file.is_file() and 'snapshots' in file.parts:
            manifest.append({'path':str(file.relative_to(DATA)), 'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
src=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
dst=DATA/'cache/torch/hub/checkpoints'/src.name
dst.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(src,dst)
manifest.append({'path':str(dst.relative_to(DATA)),'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()})
(DATA/'cache/manifest.json').write_text(json.dumps(manifest,indent=2))
print('Public upstream model cache copied and hashed:',len(manifest),'files')
