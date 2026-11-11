import json,subprocess,sys,time
from pathlib import Path
from explorer3.paths import DATA,offline_environment
from explorer3.training import train_command
from explorer3.resources import ResourceLease
with ResourceLease('compute'):
    root=Path(json.loads((DATA/'validation/act-pass.json').read_text())['run'])/'synthetic_dataset'
    output=DATA/'validation'/('smolvla-trainer-'+str(int(time.time())))
    command=train_command(root,output,policy='smolvla',steps=1,batch=1)
    subprocess.run(command,env=offline_environment(),check=True)
    subprocess.run([sys.executable,'-m','explorer3.reload_check',str(output/'checkpoints/last/pretrained_model'),str(root)],env=offline_environment(),check=True)
    (DATA/'validation/smolvla-trainer-pass.json').write_text(json.dumps({'path':str(output),'synthetic':True,'steps':1,'batch':1}))
