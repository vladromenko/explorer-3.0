"""ROSMASTER recording → LeRobotDataset → validator, against a fake agent (synthetic, no robot)."""
import json, os, subprocess, sys
import pytest


@pytest.mark.parametrize('name', ['rosmaster_dock', 'rosmaster_pick_place'])
def test_recording_writes_valid_causal_episode(tmp_path, name):
    env = {**os.environ, 'EXPLORER3_DATA': str(tmp_path), 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    out = subprocess.run([sys.executable, 'tests/rosmaster_fake_agent.py', name], env=env, capture_output=True,
                         text=True, timeout=180, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result['frames'] >= 25 and result['status']['phase'] == 'idle'
    rows = [json.loads(l) for l in next(tmp_path.glob('datasets/*/*/research.jsonl')).read_text().splitlines()]
    first = rows[0]
    assert first['action_send_start_ns'] >= first['observation_timestamp_ns'] > first['camera']['timestamp_ns']
    assert len(first['observation.state']) == (6 if name.endswith('dock') else 6)
    assert len(first['action']) == (3 if name.endswith('dock') else 6)
    assert first['measured_response_kind'] == ('wheel_encoder_velocity' if name.endswith('dock') else 'servo_position_readback')


def test_pilot_assignment_hold_correction_and_segments(tmp_path):
    env = {**os.environ, 'EXPLORER3_DATA': str(tmp_path), 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    out = subprocess.run([sys.executable, 'tests/rosmaster_fake_agent.py', 'rosmaster_pick_place', 'pilot'], env=env,
                         capture_output=True, text=True, timeout=240, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result['assigned'] == 1 and result['outcome'] == 'success'
    assert result['events'] == ['request', 'operator_notified', 'human_start', 'human_end', 'finish']
    assert result['categories'] == ['rollout', 'correction', 'rollout']
    assert result['features']['response_change'] is not None  # measured servo readback feeds physical features
