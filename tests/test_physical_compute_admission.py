"""Actual process locks, no robot imports or commands."""
import pytest
from explorer3 import resources,catalog,control
from explorer3.resources import ResourceLease,ResourceBusy


def test_two_manual_robots_share_activity_but_exclude_heavy_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'DATA',tmp_path)
    with ResourceLease('physical-activity',shared=True):
        with ResourceLease('physical-activity',shared=True):
            with pytest.raises(ResourceBusy):
                with ResourceLease('physical-activity'):pytest.fail('Heavy compute admitted during physical activity')
    with ResourceLease('physical-activity'):
        with pytest.raises(ResourceBusy):
            with ResourceLease('physical-activity',shared=True):pytest.fail('Motion admitted during heavy compute')


def test_manual_stop_releases_compute_exclusion(tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'DATA',tmp_path);monkeypatch.setattr(catalog,'event',lambda *a:None)
    c=control.Control();monkeypatch.setattr(c,'authorized',lambda:True)
    c.acquire('fixture-browser')
    with pytest.raises(ResourceBusy):
        with ResourceLease('physical-activity'):pass
    c.stop('fixture STOP');c.stop('repeat STOP')
    with ResourceLease('physical-activity'):pass
    c.acquire('new-browser');assert c.owner=='new-browser'
    c.stop('done')
    generation=c.generation;c.stop('late STOP')
    with pytest.raises(PermissionError):c.acquire('stale-browser',expected_generation=generation)
    assert c.owner is None
