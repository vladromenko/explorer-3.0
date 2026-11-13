"""Training command preparation only; never execute optimizer steps."""
import json
from explorer3.training import train_command
from explorer3 import policy_catalog


def test_smolvla_uses_pinned_base_and_finetune_does_not_duplicate_path(tmp_path,monkeypatch):
    dataset=tmp_path/'dataset';(dataset/'meta').mkdir(parents=True)
    (dataset/'meta/info.json').write_text(json.dumps({'features':{
        'observation.state':{'shape':[2]},'observation.images.front':{'shape':[400,640,3]},'action':{'shape':[2]}}}))
    monkeypatch.setattr(policy_catalog,'smolvla_paths',lambda:(tmp_path/'base',tmp_path/'vlm'))
    for initial in (None,str(tmp_path/'trained')):
        command=train_command(dataset,tmp_path/'output','smolvla',steps=2000,batch=1,initial_checkpoint=initial)
        paths=[arg for arg in command if arg.startswith('--policy.path=')]
        assert len(paths)==1 and paths[0]=='--policy.path='+str(initial or tmp_path/'base')
        assert '--policy.type=smolvla' not in command
        feature=json.loads(next(x.split('=',1)[1] for x in command if x.startswith('--policy.input_features=')))
        assert feature['observation.images.front']['shape']==[3,400,640]
        assert '--steps=2000' in command and '--policy.push_to_hub=false' in command


def test_diffusion_invokes_official_trainer():
    from pathlib import Path
    command=train_command(Path('/tmp/data'),Path('/tmp/out'),'diffusion',steps=2000,batch=2)
    assert '--policy.type=diffusion' in command and 'lerobot.scripts.lerobot_train' in command
