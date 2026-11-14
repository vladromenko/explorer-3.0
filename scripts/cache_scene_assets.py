"""Fetch only pinned robosuite object assets; runtime never downloads models."""
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
FILES=['LICENSE','robosuite/models/assets/objects/bottle.xml','robosuite/models/assets/objects/lemon.xml',
    'robosuite/models/assets/objects/can.xml','robosuite/models/assets/objects/meshes/bottle.stl',
    'robosuite/models/assets/objects/meshes/lemon.msh','robosuite/models/assets/objects/meshes/can.msh',
    'robosuite/models/assets/textures/glass.png','robosuite/models/assets/textures/lemon.png','robosuite/models/assets/textures/soda.png']
root=ROOT/'explorer3/sim/vendor_props';hashes={}
for name in FILES:
    relative=name.removeprefix('robosuite/models/assets/')
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/{REVISION}/{name}',timeout=30) as response:raw=response.read()
    path.write_bytes(raw);hashes[relative]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
(root/'manifest.json').write_text(json.dumps({'repository':'ARISE-Initiative/robosuite','revision':REVISION,
    'license':'MIT; see LICENSE','files':hashes,'reuse':'source meshes and textures, adapted dynamics in room.py'},indent=2))
print(json.dumps({'files':len(hashes),'bytes':sum(x['bytes'] for x in hashes.values())}))
