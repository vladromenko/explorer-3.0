"""Read an actual finalized episode; never open hardware or create samples."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from explorer3.paths import DATA
from explorer3 import catalog
from .timing import validate_timing


def validate(episode_id: str, *, domain='real'):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rows=[e for e in catalog.rows('episodes') if e['id']==episode_id]
    if len(rows)!=1:
        raise ValueError('Finalized episode not found')
    entry=rows[0]
    root=Path(entry['dataset']).resolve()
    if not root.is_relative_to(DATA/'datasets'):
        raise ValueError('Dataset outside Explorer storage')
    provenance=json.loads((root/'provenance.json').read_text())
    if domain not in {'real','sim','fixture'}:
        raise ValueError('Unknown validation domain')
    if provenance.get('domain','real')!=domain or provenance.get('synthetic') is not (domain!='real'):
        raise ValueError('Recording domain mismatch')
    if domain=='sim' and not all(k in provenance['hardware'] for k in ('model','seed','controller','lineage')):
        raise ValueError('Missing simulation provenance')
    cfg=provenance['profile']
    records=[json.loads(line) for line in (root/'research.jsonl').read_text().splitlines()]
    ds=LeRobotDataset('local/physical_readback',root=root,video_backend='pyav')
    if len(ds)!=len(records) or len(ds)!=entry['frames'] or not records:
        raise ValueError('Frame/sidecar/catalogue mismatch')
    previous=-1
    sensor_index={}
    cached_sensor=None
    cached_sensor_ref=None
    if cfg.get('environment_channels'):
        sensor_index={s['file']:s for s in (json.loads(line) for line in (root/'sensors/index.jsonl').read_text().splitlines())}
    for index,row in enumerate(records):
        if row['frame_index']!=index or row['episode_id']!=episode_id:
            raise ValueError('Invalid sidecar linkage')
        obs=row['observation_timestamp_ns']
        if not previous<obs:
            raise ValueError('Noncausal command/observation')
        validate_timing(row)
        if row['camera']['timestamp_ns']>obs:
            raise ValueError('Future camera data')
        if domain=='real' and cfg['robot']=='tank':
            if row['measured_response_kind']!='previous_command_estimate_no_wheel_feedback':
                raise ValueError('Tank command estimate mislabeled as measurement')
            if row['agent_accepted_command'] is not True or row['battery_voltage']<6.9:
                raise ValueError('Unaccepted command or low battery')
        previous=obs
        if row['action_acknowledged'] is not False:
            raise ValueError('Unexpected claimed acknowledgement')
        if not all(math.isfinite(v) for v in row['action']+row['observation.state']):
            raise ValueError('Nonfinite action/state')
        if provenance.get('training_label'):
            from .noise_demonstrations import training_target
            training_target(row,domain=domain,label=provenance['training_label'])
        if cfg.get('environment_channels'):
            from .sensors import decode_packet,environment_features
            ref=(root/row['sensor_packet']).resolve()
            if not ref.is_relative_to(root/'sensors'):raise ValueError('Sensor packet outside episode')
            entry_sensor=sensor_index.get(ref.name)
            if entry_sensor is None or entry_sensor['homebrain_received_ns']>obs:raise ValueError('Missing/future sensor receipt')
            if cached_sensor_ref!=ref:
                cached_sensor=decode_packet(ref.read_bytes(),entry_sensor['homebrain_received_ns'],entry_sensor['request_elapsed_s'])
                cached_sensor_ref=ref
            expected=environment_features(cached_sensor,obs)
            if cached_sensor['meta']['robot']!=cfg['robot']:raise ValueError('Wrong robot sensor packet')
            if not np.allclose(expected,row['observation.environment_state']):raise ValueError('Sensor feature mismatch')
    for index in sorted({0,len(ds)//2,len(ds)-1}):
        frame=ds[index]
        shape=cfg.get('image_shape',[480,640,3])
        assert tuple(frame['observation.images.'+cfg['camera']].shape)==(shape[2],shape[0],shape[1])
        from .noise_demonstrations import training_target
        target=training_target(records[index],domain=domain,label=provenance.get('training_label'))
        assert np.allclose(frame['action'].numpy(),target,atol=1e-5)
        assert np.allclose(frame['observation.state'].numpy(),records[index]['observation.state'],atol=1e-5)
        if cfg.get('environment_channels'):
            assert np.allclose(frame['observation.environment_state'].numpy(),records[index]['observation.environment_state'],atol=1e-5)
    times=np.array([r['observation_timestamp_ns'] for r in records],dtype=np.int64)
    intervals=np.diff(times)/1e6
    sequence=[r['camera']['sequence'] for r in records]
    result={'episode_id':episode_id,'outcome':entry['outcome'],'frames':len(ds),'readback':'PASS',
            'synthetic':domain!='real','domain':domain,'observation_action_availability_order':'PASS',
            'observation_action_causality':'availability_only' if any(r.get('action_time_kind')=='command_in_force_at_next_agent_snapshot' for r in records) else 'PASS','video_sampled_first_middle_last':'PASS',
            'duration_seconds':float((times[-1]-times[0])/1e9),
            'actual_hz':float((len(times)-1)*1e9/(times[-1]-times[0])) if len(times)>1 else None,
            'interval_ms_p50_p95_max':np.percentile(intervals,[50,95,100]).tolist() if len(intervals) else [],
            'reused_camera_frames':sum(a==b for a,b in zip(sequence,sequence[1:])),
            'camera_age_ms_p95':float(np.percentile([(r['observation_timestamp_ns']-r['camera']['timestamp_ns'])/1e6 for r in records],95)),
            'units':provenance['profile']['units'],'channels':provenance['profile']['channels'],
            'physical_policy_skill_verified':False}
    if domain=='real' and cfg['robot']=='tank':
        result.update(camera_age_ms_p95_meaning='Homebrain receive-to-observation; not camera capture age',
            camera_age_at_agent_ms_p95=float(np.percentile([r['camera']['age_at_agent_response_ns']/1e6 for r in records],95)),
            paused_frames=sum(bool(r.get('motion_paused')) for r in records),
            nonzero_command_frames=sum(any(abs(a)>1e-6 for a in r['action']) for r in records),
            measured_wheel_feedback=False)
    from .datasets import fingerprint
    result['dataset_sha256']=fingerprint(root)
    (DATA/'validation'/f'episode-{episode_id}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episode_id')
    validate(parser.parse_args().episode_id)
