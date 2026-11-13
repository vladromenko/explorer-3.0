"""Actual frozen ACT feature/chunk/snapshot checks; no training or hardware."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from explorer3.sim.evaluate import ACTClient
from explorer3.profiles import profile


def check(checkpoint,spool,output):
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((spool/'manifest.json').read_text())
    lines=(spool/'samples.jsonl').read_text().splitlines()
    sample=json.loads(lines[min(30,len(lines)-1)])
    obs={'state':np.asarray(sample['observation.state']),
         'rgb':cv2.cvtColor(cv2.imread(str(spool/sample['image_path'])),cv2.COLOR_BGR2RGB)}
    client=ACTClient(checkpoint,output/'worker.log')
    try:
        cfg=profile(meta['profile']);client.action(obs,cfg);before=client.request('snapshot')
        proposal=client.observation_request('propose',obs,cfg);after=client.request('snapshot')
        assert before==after,'Scoring mutated execution state'
        expected=[client.action(obs,cfg) for _ in range(3)]
        client.request('restore',snapshot=before)
        assert expected==[client.action(obs,cfg) for _ in range(3)],'Queue restore differs'
        client.request('reset');fresh=client.action(obs,cfg)
        assert np.allclose(fresh,proposal['chunk'][0]),'Proposal differs from executed ACT action'
        result={'actual_official_ACT':True,'existing_checkpoint':str(checkpoint),'policy_id':proposal['policy_id'],
            'feature_shape':list(np.asarray(proposal['features']).shape),'chunk_shape':list(np.asarray(proposal['chunk']).shape),
            'proposal_leaves_queue_unchanged':'PASS','snapshot_restore_actions':'PASS','proposal_matches_actual_first_action':'PASS',
            'training_started':False,'source':'stored image, inference plumbing only; no policy skill evaluation'}
        (output/'result.json').write_text(json.dumps(result,indent=2))
        (output/'proposal.json').write_text(json.dumps(proposal))
        print(json.dumps(result),flush=True)
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--spool',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();check(args.checkpoint,args.spool,args.output)
