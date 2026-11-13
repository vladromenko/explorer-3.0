import pytest
from explorer3 import fleet_runs


def test_fixture_config_is_explicit_and_has_no_training(monkeypatch):
    monkeypatch.setattr(fleet_runs.catalog,'rows',lambda *_:[])
    cfg=fleet_runs.prepare({'so101':'fixture','tank':'fixture'})
    assert not cfg['training_started'] and cfg['kind']=='fleet_execution'
    assert all(x['kind']=='fixture' for x in cfg['specs'].values())
    for kwargs in ({'horizon_s':float('nan')},{'horizon_s':100},{'help_s':0}):
        with pytest.raises(ValueError):fleet_runs.prepare({'tank':'fixture'},**kwargs)
    with pytest.raises(ValueError):fleet_runs.prepare({'tank':'missing'})
