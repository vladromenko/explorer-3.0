"""Install only original M3Pro model files from the pinned public vendor archive.

Normal Explorer startup is offline. --download is a one-time setup action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
from explorer3.paths import DATA

SHA256='7f9083d301fef50633445272505e9acb561a4b6abe362dd946d7deaaf4caff0f'
URL='https://drive.usercontent.google.com/download?id=1WxZbB2ulfWldWYRKiT5zheso-8EtqfdV&export=download&confirm=t'


def install(download=False):
    root=DATA/'cache/source/yahboom-downloads';root.mkdir(parents=True,exist_ok=True)
    archive=root/'m3pro-code.zip'
    if not archive.exists():
        if not download:raise FileNotFoundError('Original archive missing; run with --download for first setup')
        partial=archive.with_suffix('.partial')
        with urllib.request.urlopen(URL,timeout=60) as source,partial.open('wb') as destination:
            shutil.copyfileobj(source,destination,4*1024*1024)
        partial.rename(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=SHA256:raise ValueError('Vendor archive differs from pinned source; audit before updating')
    destination=root/'m3pro-orin';files={}
    choices={'orin/M3Pro_ws-0.0.1.zip':lambda p:p.startswith('m3pro_ws/src/M3Pro/') and p.endswith(('.STL','.urdf','.xml')),
        'orin/yahboomcar_ws.zip':lambda p:p=='yahboomcar_ws/src/M3Pro_config/config/M3Pro.srdf'}
    with zipfile.ZipFile(archive) as outer,tempfile.TemporaryDirectory(prefix='explorer-source-') as directory:
        for name,selected in choices.items():
            inner_path=Path(directory)/'inner.zip'
            with outer.open(name) as source,inner_path.open('wb') as target:shutil.copyfileobj(source,target,4*1024*1024)
            with zipfile.ZipFile(inner_path) as inner:
                for entry in inner.infolist():
                    if selected(entry.filename) and not entry.is_dir():
                        relative=Path(entry.filename)
                        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source archive path')
                        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        blob=inner.read(entry);target.write_bytes(blob);files[str(relative)]=hashlib.sha256(blob).hexdigest()
    manifest={'source_url':URL,'archive_sha256':SHA256,'files':files,'retrieved':'2026-11-13',
        'license_note':'package.xml declares Apache 2.0 with stale TurtleBot attribution; assets kept in local cache'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    print(json.dumps({'cached_files':len(install(parser.parse_args().download)['files'])}))
