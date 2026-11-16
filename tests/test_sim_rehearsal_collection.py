"""A physics revision must not reuse old successful demonstrations as its collection budget."""
from types import SimpleNamespace

import pytest

from scripts import sim_rehearsal as rehearsal


def test_collection_counts_only_current_physics_and_limits_workers(monkeypatch):
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
    assert rehearsal.collect('rosmaster_floor_pick_knowledge_sim', 3, workers=1) == 3
    assert len(launches) == 1


def test_failed_collection_cannot_silently_continue_to_training(monkeypatch):
    monkeypatch.setattr(rehearsal.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='a' * 64))
    monkeypatch.setattr(rehearsal, 'successes', lambda *a: [])
    monkeypatch.setattr(rehearsal.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(wait=lambda: 1))
    with pytest.raises(RuntimeError, match='обучение не запускается'):
        rehearsal.collect('rosmaster_floor_pick_knowledge_sim', 3, workers=1)
