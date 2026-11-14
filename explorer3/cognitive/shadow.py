"""ROSMASTER archive review, with no model, network or execution capability.

This prepares the observation portion of AG3. Archive arrival metadata describes
the recording instant only; reading it now never creates a live observation.
"""
import math
import time
from . import store
from .schemas import ObservationEvidence

VERSION='rosmaster-archive-review-v1'
STREAMS={'rgb':'Камера','depth':'Глубина','scan':'Передний лидар',
         'scan_rear':'Задний лидар','imu':'Инерциальный датчик','odometry':'Одометрия'}


def summarize(value,*,now=None):
    item=ObservationEvidence.model_validate(value)
    if item.domain!='real' or item.robot!='rosmaster':
        raise ValueError('ROSMASTER real evidence required')
    if store.digest(item.payload)!=item.content_sha256:
        raise ValueError('Evidence payload hash mismatch')
    if item.source!='recorded_sensor_snapshot' or item.payload.get('archived') is not True:
        raise ValueError('Only original archived sensor evidence is admitted here')
    now=time.time() if now is None else now
    if not math.isfinite(now):raise ValueError('Finite review time required')
    rows=[]
    streams=item.payload.get('streams',{})
    for name,label in STREAMS.items():
        meta=streams.get(name)
        age=meta.get('age_s') if isinstance(meta,dict) else None
        valid_age=isinstance(age,(int,float)) and not isinstance(age,bool) and math.isfinite(age) and age>=0
        state='missing' if meta is None else 'unknown'
        if valid_age:state='reported_recent_at_capture' if age<=2 else 'reported_stale_at_capture'
        rows.append({'stream':name,'label':label,'capture_status':state,
            'reported_capture_age_s':age if valid_age else None,
            'frame_id':meta.get('frame_id') if isinstance(meta,dict) else None})
    return {'version':VERSION,'robot':'rosmaster','mode':'archive_review',
        'evidence_ids':[item.id],'content_sha256':item.content_sha256,
        'artifact_sha256':item.artifact_sha256,'reviewed_at':now,
        'source_clock':{'domain':item.clock_domain,'boot_id':item.boot_id,
                        'original_clock_pair':item.payload.get('clock_pair')},
        'streams':rows,'live_observation':False,'hardware_readiness':'unknown',
        'proposed_next_step':'review_missing_streams' if any(r['capture_status']!='reported_recent_at_capture' for r in rows)
                             else 'request_new_read_only_observation_when_authorized',
        'motion_admitted':False,'model_called':False,'training_started':False,
        'helper_outcome':'unknown','recipient_outcome':'unknown','task_outcome':'unknown',
        'limitations':['metadata ages do not validate array values or synchronization',
                      'archive cannot establish present robot state',
                      'raw servo angles are not calibrated joint coordinates',
                      'shared world transform and physical outcomes remain unverified']}


def review(evidence_id):
    # Reuse immutable store validation, including the original file SHA. Old
    # evidence is permitted for archive review only, never for skill admission.
    item=store.evidence([evidence_id],domain='real',fresh=False)[0]
    if not item.artifact_ref or not item.artifact_sha256:
        raise ValueError('Original archived packet and its SHA are required')
    store.validate_evidence_artifact(item)
    result=summarize(item)
    store.event(None,'rosmaster_archive_review',result)
    return result
