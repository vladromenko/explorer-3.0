"""Explicit developer setup; ordinary runtime only reads the pinned local files."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from explorer3.paths import ROOT,DATA

cfg=json.loads((ROOT/'configs/cognitive-model.json').read_text())
root=DATA/'cache/cognitive/qwen3-vl-2b'
path=Path(snapshot_download(cfg['repository'],revision=cfg['revision'],local_dir=root,
    allow_patterns=['*.json','*.safetensors','*.txt','README.md','LICENSE*']))
hashes={}
for source in sorted(path.iterdir()):
    if source.is_file() and source.name!='explorer-manifest.json':
        sha=hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):sha.update(block)
        hashes[source.name]=sha.hexdigest()
assert 'model.safetensors' in hashes
manifest={**cfg,'files':hashes,'weights_changed_by_training':False}
(path/'explorer-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'model':cfg['repository'],'revision':cfg['revision'],'files':len(hashes),'path':str(path)}))
