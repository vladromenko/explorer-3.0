"""Import existing sensor archives without copying video or relabelling domains."""
import hashlib
import time
import uuid
from pathlib import Path
from explorer3.paths import DATA
from explorer3.sensors import decode_packet
from . import store
from .schemas import ObservationEvidence


def import_sensor_packet(relative):
    path=(DATA/relative).resolve()
    if not path.is_relative_to(DATA.resolve()) or not path.is_file():raise ValueError('Local archived packet required')
    raw=path.read_bytes();packet=decode_packet(raw);meta=packet['meta']
    system_ns=meta.get('clock_pair',{}).get('system_time_ns')
    created=system_ns/1e9 if system_ns else path.stat().st_mtime
    payload={'stream_names':sorted(meta['streams']),'streams':meta['streams'],
        'conditions':{},'archived':True,'clock_pair':meta.get('clock_pair'),
        'common_xyz_available':False,'positions_are_local':True}
    item=ObservationEvidence(id=uuid.uuid4().hex,domain='real',robot=meta['robot'],source='recorded_sensor_snapshot',
        sample_id=str(meta['monotonic_ns']),frame_id=meta['robot']+'/local_sensor_frames',
        received_timestamp_ns=meta['monotonic_ns'],clock_domain=meta['robot']+'_monotonic',
        boot_id=meta.get('boot_id'),created_at=created,expires_at=created+2,
        status='observed',quality={'archive_import':True,'freshness_measured_from_original_time':bool(system_ns)},
        artifact_ref=str(path.relative_to(DATA.resolve())),artifact_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256=store.digest(payload),payload=payload)
    return store.put_evidence(item)
