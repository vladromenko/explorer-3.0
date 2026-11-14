"""Archive legacy admission files; no services or devices are started/stopped."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explorer3 import permission_migration
from explorer3.paths import DATA

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=DATA)
parser.add_argument('--rollback',type=Path)
args=parser.parse_args()
result=(permission_migration.rollback(args.data,args.rollback) if args.rollback
        else permission_migration.migrate(args.data))
print(json.dumps(result,ensure_ascii=False,indent=2))
