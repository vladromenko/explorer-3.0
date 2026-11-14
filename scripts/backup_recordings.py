#!/usr/bin/env python3
"""Explicit local backup/verification/restore. Never overwrite the active catalogue."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explorer3 import data_safety
from explorer3.paths import DATA

parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['backup','verify','restore'])
parser.add_argument('--backup',type=Path)
parser.add_argument('--episodes-root',type=Path,default=DATA)
parser.add_argument('--destination',type=Path)
args=parser.parse_args()
if args.action=='backup':result=data_safety.backup_now(force=True)
else:
    if not args.backup:parser.error('--backup is required')
    if args.action=='restore':
        if not args.destination:parser.error('--destination must be a NEW SQLite file')
        result=data_safety.restore_catalog(args.backup,args.destination)
    else:
        manifest=json.loads((args.backup/'manifest.json').read_text())
        assert data_safety.sha256(args.backup/'catalog.sqlite3')==manifest['catalog_sha256'],'catalog hash mismatch'
        for relative,entry in manifest['files'].items():
            path=(args.episodes_root/relative).resolve()
            assert path.is_relative_to(args.episodes_root.resolve()),'invalid manifest path'
            assert data_safety.sha256(path)==entry['sha256'],relative
        result={'verified_files':len(manifest['files']),'catalog':'PASS','episodes':'PASS'}
print(json.dumps(result,ensure_ascii=False,indent=2))
