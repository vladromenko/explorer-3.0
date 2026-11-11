"""Read an actual finalized episode; never open hardware or create samples."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from explorer3.paths import DATA
from explorer3 import catalog


def validate(episode_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rows=[e for e in catalog.rows('episodes') if e['id']==episode_id]
    if len(rows)!=1:
        raise ValueError('Finalized episode not found')
    entry=rows[0]
    root=Path(entry['dataset']).resolve()
    if not root.is_relative_to(DATA/'datasets'):
        raise ValueError('Dataset outside Explorer storage')
    provenance=json.loads((root/'provenance.json').read_text())
    if provenance.get('synthetic') is not False:
        raise ValueError('Not a real physical recording')
    records=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
    ds=LeRobotDataset('local/physical_readback',root=root,video_backend='pyav')
    if len(ds)!=len(records) or len(ds)!=entry['frames'] or not records:
        raise ValueError('Frame/sidecar/catalogue mismatch')
    previous=-1
    for index,row in enumerate(records):
        if row['frame_index']!=index or row['episode_id']!=episode_id:
            raise ValueError('Invalid sidecar linkage')
        obs=row['observation_timestamp_ns']
        if not previous<obs<=row['action_send_start_ns']<=row['action_send_end_ns']:
            raise ValueError('Noncausal command/observation')
        if row['camera']['timestamp_ns']>obs:
            raise ValueError('Future camera data')
        previous=obs
        if row['action_acknowledged'] is not False:
            raise ValueError('Unexpected claimed acknowledgement')
        if not all(math.isfinite(v) for v in row['action']+row['observation.state']):
            raise ValueError('Nonfinite action/state')
    for index in sorted({0,len(ds)//2,len(ds)-1}):
        frame=ds[index]
        assert tuple(frame['observation.images.wrist'].shape)==(3,480,640)
        assert np.allclose(frame['action'].numpy(),records[index]['action'],atol=1e-5)
        assert np.allclose(frame['observation.state'].numpy(),records[index]['observation.state'],atol=1e-5)
    times=np.array([r['observation_timestamp_ns'] for r in records],dtype=np.int64)
    intervals=np.diff(times)/1e6
    sequence=[r['camera']['sequence'] for r in records]
    result={'episode_id':episode_id,'outcome':entry['outcome'],'frames':len(ds),'readback':'PASS',
            'synthetic':False,'observation_action_causality':'PASS','video_sampled_first_middle_last':'PASS',
            'duration_seconds':float((times[-1]-times[0])/1e9),
            'actual_hz':float((len(times)-1)*1e9/(times[-1]-times[0])) if len(times)>1 else None,
            'interval_ms_p50_p95_max':np.percentile(intervals,[50,95,100]).tolist() if len(intervals) else [],
            'reused_camera_frames':sum(a==b for a,b in zip(sequence,sequence[1:])),
            'camera_age_ms_p95':float(np.percentile([(r['observation_timestamp_ns']-r['camera']['timestamp_ns'])/1e6 for r in records],95)),
            'units':provenance['profile']['units'],'channels':provenance['profile']['channels'],
            'physical_policy_skill_verified':False}
    (DATA/'validation'/f'episode-{episode_id}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episode_id')
    validate(parser.parse_args().episode_id)
