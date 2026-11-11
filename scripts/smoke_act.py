"""Synthetic software fixture only. No episodes are inserted into the real catalogue."""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from explorer3.paths import DATA,offline_environment
from explorer3.training import train_command
from explorer3.resources import ResourceLease
from lerobot.datasets.lerobot_dataset import LeRobotDataset

with ResourceLease('compute'):
    run=DATA/'validation'/('act-'+str(int(time.time())))
    run.mkdir()
    root=run/'synthetic_dataset'
    features={'observation.state':{'dtype':'float32','shape':(6,),'names':[str(i) for i in range(6)]},
              'action':{'dtype':'float32','shape':(6,),'names':[str(i) for i in range(6)]},
              'observation.images.wrist':{'dtype':'video','shape':(64,64,3),'names':['height','width','channels']}}
    ds=LeRobotDataset.create(repo_id='local/synthetic_software_only',root=root,features=features,fps=30,
                            robot_type='synthetic_fixture',use_videos=True,video_backend='pyav',image_writer_threads=2)
    rng=np.random.default_rng(1000)
    for episode in range(3):
        for frame in range(12):
            ds.add_frame({'observation.state':rng.normal(size=6).astype('float32'),
                          'action':rng.normal(size=6).astype('float32'),
                          'observation.images.wrist':rng.integers(0,256,size=(64,64,3),dtype='uint8'),
                          'task':'Synthetic software fixture; never execute on a robot.'})
        ds.save_episode()
    ds.finalize()
    reread=LeRobotDataset('local/synthetic_software_only',root=root,video_backend='pyav')
    assert len(reread)==36
    output=run/'training'
    command=train_command(root,output,steps=50,batch=8)
    (run/'command.json').write_text(json.dumps(command,indent=2))
    (run/'DISCLAIMER.json').write_text(json.dumps({'synthetic':True,'hardware_opened':False,'physical_pass':False,'image_size':[64,64]}))
    subprocess.run(command,env=offline_environment(),check=True)
    checkpoint=output/'checkpoints/last/pretrained_model'
    subprocess.run([sys.executable,'-m','explorer3.reload_check',str(checkpoint),str(root)],env=offline_environment(),check=True)
    (DATA/'validation/act-pass.json').write_text(json.dumps({'run':str(run),'synthetic':True,'steps':50,'physical_skill_verified':False}))
    print('SOFTWARE PASS:',run,flush=True)
