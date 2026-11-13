"""Pinned local model assets. Hash verification never downloads at inference time."""
import hashlib
import json
from pathlib import Path
from .paths import ROOT


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def verify(root,method):
    manifest=json.loads((ROOT/'configs/vision-assets.json').read_text())
    if method=='yoloe':base=Path(root);entries=manifest['weights']
    elif method=='sam3':base=Path(root)/'sam3';entries=manifest['sam3']['files']
    else:raise ValueError('Unknown vision model')
    for entry in entries:
        path=base/entry['file']
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('Модель отсутствует или изменилась: '+str(path))
    return {'manifest_sha256':digest(ROOT/'configs/vision-assets.json'),'files':entries}
