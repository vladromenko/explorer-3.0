"""ROSMASTER recording → LeRobotDataset → validator, against a fake agent (synthetic, no robot)."""
import json, os, subprocess, sys
import pytest


@pytest.mark.parametrize('phase,frames', [('preparing', 0), ('recording', 0), ('saving', 10)])
def test_outcome_requires_captured_frames(monkeypatch, phase, frames):
    from explorer3.rosmaster_session import RosmasterSession
    session = RosmasterSession(object())
    monkeypatch.setattr(session, 'active', lambda: True)
    session.state.update(phase=phase, frames=frames)
    session.emergency = False
    for outcome in ('success', 'failure'):
        with pytest.raises(RuntimeError, match='счётчика кадров'):
            session.finish(outcome)
        assert not session.stop_event.is_set()
        assert session.outcome == 'aborted'
    assert session.finish('aborted')  # cancellation remains available while preparing


def test_completed_outcome_cannot_be_overwritten(monkeypatch):
    from explorer3.rosmaster_session import RosmasterSession
    session = RosmasterSession(object())
    monkeypatch.setattr(session, 'active', lambda: True)
    session.emergency = False
    session.state.update(phase='recording', frames=10)
    assert session.finish('success')
    assert not session.finish('failure')
    assert session.outcome == 'success'
    assert session.termination == 'operator_success'


def test_idle_finish_cannot_invent_success():
    from explorer3.rosmaster_session import RosmasterSession
    session = RosmasterSession(object())
    assert not session.finish('success')
    assert session.outcome == 'aborted'


def test_cancel_during_writer_setup_reports_no_saved_episode(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from explorer3 import rosmaster_session as rs
    session = rs.RosmasterSession(object())
    monkeypatch.setattr(session, 'active', lambda: True)
    session.emergency = False
    session.termination = None
    session.study = None
    session.state.update(phase='preparing')
    monkeypatch.setattr(rs, 'ResourceLease', lambda name: nullcontext())
    monkeypatch.setattr(rs, 'agent_client', lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(rs.ClockMap, 'probe', lambda self, count=5: {})
    finalized = []
    class ColdWriter:
        def __init__(self, *args, **kwargs):
            session.finish('aborted')
        def finish(self, outcome, provenance):
            finalized.append(outcome)
    monkeypatch.setattr(rs, 'EpisodeWriter', ColdWriter)
    session.run('test', 'rosmaster_dock', 'scene', 'demonstration', 'test', True)
    assert finalized == ['aborted']
    assert session.status()['phase'] == 'idle'
    assert session.status()['outcome'] == 'aborted'
    assert session.status()['segments'] == []
    assert 'до первого кадра' in session.status()['error']


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
    assert first['action_send_start_ns'] is None and first['action_send_end_ns'] is None
    assert first['action_available_ns'] >= first['observation_timestamp_ns'] > first['camera']['timestamp_ns']
    from explorer3.timing import send_latency_s
    assert send_latency_s(first) is None
    assert len(first['observation.state']) == (6 if name.endswith('dock') else 6)
    assert len(first['action']) == (3 if name.endswith('dock') else 6)
    assert first['measured_response_kind'] == ('wheel_encoder_velocity' if name.endswith('dock') else 'servo_position_readback')
    provenance = json.loads(next(tmp_path.glob('datasets/*/*/provenance.json')).read_text())['hardware']
    assert provenance['termination_reason'] == 'operator_success' and provenance['domain'] == 'real'
    assert provenance['source'] == 'human' and len(provenance['agent_code_sha256']) == 4


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


def test_console_trial_and_record_share_collection_and_scene():
    from pathlib import Path
    page=(Path(__file__).resolve().parents[1]/'explorer3/static/rosmaster-console.html').read_text()
    import re
    for action in ('trial_start','record_start'):
        payload=re.search(r"send\(\{action:'"+action+r"'.*?\}\)",page).group()
        assert "collection:$('rcollection').value" in payload
        assert "scene:$('rscene').value" in payload
