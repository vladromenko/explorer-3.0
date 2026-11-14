#!/usr/bin/env python3
"""Install only a verified Explorer payload. No service or hardware commands."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid

LEGACY=('incident-block.json','bench-authorized.json','authorization-suspended-incident.json','physical-block-authorized.json')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def target(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative)!=name:raise ValueError('Unsafe payload path')
    destination=root/name
    if not destination.resolve().is_relative_to(root.resolve()):raise ValueError('Payload escapes Explorer root')
    if name in ('config.json','camera.yaml') or name.startswith(('calibration/','vendor/')):raise ValueError('Protected configuration')
    return destination


def install(stage,root):
    stage=Path(stage);root=Path(root)
    manifest=json.loads((stage/'MANIFEST.json').read_text())
    if manifest['platform']!='tank':raise ValueError('Not a Tank bundle')
    files={name[len('payload/'):]:checksum for name,checksum in manifest['files'].items() if name.startswith('payload/')}
    if not files or any(name in LEGACY for name in files):raise ValueError('Payload must not grant or restore legacy permissions')
    for name,checksum in files.items():
        target(root,name)
        if sha(stage/'payload'/name)!=checksum:raise ValueError('Staging hash mismatch: '+name)
    backup=root/'archive'/('deploy-recording-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True,exist_ok=False)
    originals={}
    for name in [*files,*LEGACY]:
        source=target(root,name)
        originals[name]={'exists':source.exists(),'sha256':sha(source) if source.exists() else None}
        if source.exists():
            dest=backup/'files'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    report={'status':'prepared','root':str(root.resolve()),'files':files,'originals':originals,
        'reason':'RECORDING_READINESS_PROMPT_RU R2, operator floor authorization; no motion invoked',
        'archive':str(backup.resolve()),'created_at':time.time()}
    journal=backup/'deployment.json';journal.write_text(json.dumps(report,indent=2))
    try:
        for name in files:
            dest=target(root,name);dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_name(dest.name+'.deploy-tmp');shutil.copy2(stage/'payload'/name,temporary);temporary.replace(dest)
        for name,checksum in files.items():
            if sha(target(root,name))!=checksum:raise ValueError('Installed hash mismatch: '+name)
        for name in LEGACY:target(root,name).unlink(missing_ok=True)
        report['status']='installed';journal.write_text(json.dumps(report,indent=2))
        return report
    except Exception:
        rollback(backup)
        raise


def rollback(backup):
    backup=Path(backup);report=json.loads((backup/'deployment.json').read_text());root=Path(report['root'])
    for name,item in report['originals'].items():
        if item['exists'] and sha(backup/'files'/name)!=item['sha256']:raise ValueError('Backup is corrupted')
    for name,item in report['originals'].items():
        dest=target(root,name)
        if item['exists']:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(backup/'files'/name,dest)
        else:dest.unlink(missing_ok=True)
    report['status']='rolled_back';(backup/'deployment.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path);parser.add_argument('--root',type=Path,default=Path('/home/ubuntu/explorer3'))
    parser.add_argument('--rollback',type=Path);args=parser.parse_args()
    if not args.stage and not args.rollback:parser.error('--stage or --rollback required')
    result=rollback(args.rollback) if args.rollback else install(args.stage,args.root)
    print(json.dumps({'status':result['status'],'archive':result['archive'],'files':len(result['files'])}))
