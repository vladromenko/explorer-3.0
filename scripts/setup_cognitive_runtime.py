"""Explicit setup, never called from the UI or runtime. Does not upgrade base libs."""
import json
import subprocess
import sys
import venv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'.venv/bin/python';target=ROOT/'.venv-cognitive'
probe=subprocess.check_output([str(base),'-c',
    'import json,site,torch,transformers,pydantic;print(json.dumps({"sites":site.getsitepackages(),"torch":torch.__version__,"transformers":transformers.__version__,"pydantic":pydantic.__version__}))'],text=True)
versions=json.loads(probe);config=json.loads((ROOT/'configs/cognitive-model.json').read_text())
if versions['transformers']!=config['transformers_version']:raise SystemExit('Base Transformers version changed; review compatibility first')
if not target.exists():venv.EnvBuilder(with_pip=True).create(target)
python=target/'bin/python'
site=Path(subprocess.check_output([str(python),'-c','import site;print(site.getsitepackages()[0])'],text=True).strip())
(site/'explorer-base.pth').write_text('\n'.join(versions['sites'])+'\n')
subprocess.run([str(python),'-m','pip','install','--no-deps','-r',str(ROOT/'configs/cognitive-requirements.lock.txt')],check=True)
print(json.dumps({'runtime':str(target),'base_versions':versions,'weights_downloaded':False,'training_started':False}))
