"""One finite existing-queue observation job, no hardware or actuation dispatcher."""
import json
import subprocess
from explorer3.paths import ROOT,DATA,offline_environment
from . import observer,store
from .run import model_proposal


def execute(config,work):
    request=config['request'];method=config['method']
    # Recheck source identities and files at actual dispatch, not only enqueue.
    observer.load_images(request)
    if method=='vlm':
        result=model_proposal(work,request)
        result.update(outcome='unknown',status='hypothesis',evidence_ids=request['evidence_ids'])
    elif method in ('yoloe','sam3'):
        spec=request['images'][0]
        interpreter=ROOT/('.venv-sim/bin/python' if method=='yoloe' else '.venv/bin/python')
        subprocess.run([str(interpreter),'-m','explorer3.perception',str(DATA/spec['artifact_ref']),str(work),
            '--method',method,'--prompt',request['question'],'--device','cuda'],
            cwd=ROOT,env=offline_environment(),check=True,timeout=100)
        measured=json.loads((work/'result.json').read_text())
        item=observer.ground_perception(spec['evidence_id'],measured)
        result={**measured,'evidence_id':item.id,'grounded':item.payload,'outcome':'unknown','status':'hypothesis'}
    else:raise ValueError('Unknown observer')
    result.update(training_started=False,motor_commands=False,hardware_access=False,
        current_robot_state='unknown',original_evidence=request['images'])
    (work/'result.json').write_text(store.encoded(result))
    store.event(None,'scene_observation',{'job_id':work.name,'method':method,
        'evidence_ids':request['evidence_ids'],'outcome':'unknown','motor_commands':False})
    return result
