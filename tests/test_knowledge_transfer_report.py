"""Resuming a MuJoCo comparison preserves completed outcomes and its frozen protocol."""
import json

import pytest

from scripts import knowledge_transfer_mujoco as experiment


def test_resume_never_repeats_completed_trials_and_rejects_changed_protocol(tmp_path, monkeypatch):
    calls = []
    def run(seed, condition):
        calls.append((seed, condition))
        return {'seed': seed, 'condition': condition, 'object': 'ball', 'success': condition == 'privileged',
                'outcome': 'success' if condition == 'privileged' else 'failure'}
    monkeypatch.setattr(experiment, 'run', run)
    argv = ['experiment', '--episodes', '1', '--start-seed', '12000', '--out', str(tmp_path)]
    monkeypatch.setattr(experiment.sys, 'argv', argv)
    experiment.main()
    assert len(calls) == 3
    original = (tmp_path / 'results.jsonl').read_bytes()
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['completed'] and summary['privileged']['success_rate'] == 1
    monkeypatch.setattr(experiment.sys, 'argv', argv + ['--resume'])
    experiment.main()
    assert len(calls) == 3 and (tmp_path / 'results.jsonl').read_bytes() == original
    monkeypatch.setattr(experiment.sys, 'argv', argv + ['--resume', '--episodes', '2'])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2


def test_partial_report_does_not_claim_completed_comparison(tmp_path):
    rows = [{'condition': 'privileged', 'object': 'bottle', 'success': True}]
    summary = experiment.save_summary(tmp_path, rows, 180)
    assert summary['completed'] is False
    assert summary['none']['n'] == 0 and summary['none']['success_rate'] is None
    assert summary['privileged']['by_object'] == {'bottle': '1/1'}
