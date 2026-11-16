"""A physics revision must not reuse old successful demonstrations as its collection budget."""
from types import SimpleNamespace

import pytest

from scripts import sim_rehearsal as rehearsal


def test_collection_counts_only_current_physics_and_limits_workers(monkeypatch, tmp_path):
    version = 'a' * 64
    counts, launches = [], []
    monkeypatch.setattr(rehearsal.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=version))
    def successes(name, model=None):
        assert model == version
        counts.append(model)
        return [] if len(counts) == 1 else [{}] * 3
    monkeypatch.setattr(rehearsal, 'successes', successes)
    def launch(args, **kw):
        launches.append(args)
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(rehearsal.subprocess, 'Popen', launch)
    assert rehearsal.collect('rosmaster_floor_pick_knowledge_sim', 3, workers=1, log_dir=tmp_path) == 3
    assert len(launches) == 1


def test_failed_collection_cannot_silently_continue_to_training(monkeypatch, tmp_path):
    monkeypatch.setattr(rehearsal.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='a' * 64))
    monkeypatch.setattr(rehearsal, 'successes', lambda *a: [])
    calls = []
    def fail(*a, **kw):
        calls.append(a)
        kw['stdout'].write('diagnostic retained\n')
        return SimpleNamespace(wait=lambda: 1)
    monkeypatch.setattr(rehearsal.subprocess, 'Popen', fail)
    with pytest.raises(RuntimeError, match='обучение не запускается'):
        rehearsal.collect('rosmaster_floor_pick_knowledge_sim', 3, workers=1, log_dir=tmp_path)
    assert len(calls) == 1  # a technical failure is not retried four times without a diagnosis
    assert next(tmp_path.glob('*.log')).read_text() == 'diagnostic retained\n'
