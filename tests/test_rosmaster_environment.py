import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

from explorer3.rosmaster_session import attach_environment,control_profile
from explorer3.sensors import SensorFeed
from rosmaster_sensor_fixture import FixtureFeed


def test_causal_history_and_missing_data():
    feed=SensorFeed(retain_history=True)
    a={'received_ns':10};b={'received_ns':20}
    feed.history.extend([a,b])
    assert feed.at_or_before(15) is a and feed.at_or_before(20) is b
    assert feed.at_or_before(9) is None
    with pytest.raises(RuntimeError,match='предшествующего'):attach_environment({'observation_ns':9},feed)


def test_environment_uses_past_and_refuses_stale_sensor_streams():
    feed=FixtureFeed(retain_history=True);feed.start()
    try:
        end=time.monotonic()+2
        while feed.latest is None and time.monotonic()<end:time.sleep(.01)
        packet=feed.latest;assert packet
        sample={'observation_ns':packet['received_ns']+10_000_000};attach_environment(sample,feed)
        assert len(sample['observation.environment_state'])==265
        assert sample['observation.environment_state'][0]==1 and sample['observation.environment_state'][178]==2
        with pytest.raises(RuntimeError,match='свежих'):
            attach_environment({'observation_ns':packet['received_ns']+3_000_000_000},feed)
    finally:feed.close()


@pytest.mark.parametrize('name',['rosmaster_dock_rgbd_lidar','rosmaster_pick_place_rgbd_lidar'])
def test_real_lerobot_roundtrip_with_environment_fixture(tmp_path,name):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'EXPLORER3_DATA':str(tmp_path),'PYTHONPATH':str(root)}
    out=subprocess.run([sys.executable,'tests/rosmaster_fake_agent.py',name],cwd=root,env=env,
                       capture_output=True,text=True,timeout=120)
    assert out.returncode==0,out.stderr[-3500:]+out.stdout[-1000:]
    result=json.loads(out.stdout.strip().splitlines()[-1]);assert result['frames']>=25
    folder=next((tmp_path/'datasets/fixture'/name).iterdir())
    provenance=json.loads((folder/'provenance.json').read_text())
    assert provenance['synthetic'] and provenance['domain']=='fixture'
    rows=[json.loads(line) for line in (folder/'research.jsonl').read_text().splitlines()]
    assert all(len(row['observation.environment_state'])==265 for row in rows)
    assert len({row['sensor_packet'] for row in rows})<len(rows)
    assert all(len(row['action'])==(3 if 'dock' in name else 6) for row in rows)
    manifest=json.loads((folder/'sensors/manifest.json').read_text())
    assert manifest['robot']=='rosmaster' and manifest['feature_schema']=='rosmaster_environment_v1'
    assert control_profile(name)==name.removesuffix('_rgbd_lidar')
