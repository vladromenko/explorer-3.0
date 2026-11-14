"""Placement contract checks prepared for the next authorized validation block."""
import json
import pytest
from explorer3.sim import layout


@pytest.fixture
def config(tmp_path,monkeypatch):
    original=json.loads((layout.ROOT/'configs/physical-layout.json').read_text())
    path=tmp_path/'configs/physical-layout.json';path.parent.mkdir()
    path.write_text(json.dumps(original));monkeypatch.setattr(layout,'ROOT',tmp_path)
    return path,original


def test_unmeasured_height_is_never_real_calibration(config):
    value=layout.placement()
    assert value['so101_support']['kind']=='table'
    assert value['so101_support']['measured_height_m'] is None
    assert value['so101_height_status']=='assumed_for_simulation'
    assert value['common_world_transform_verified'] is False
    assert value['rosmaster_support']==value['tank_support']=={'kind':'floor','height_m':0.}


def test_measurement_requires_source_and_does_not_calibrate_world(config):
    path,value=config;value['so101_support']['measured_height_m']=.8
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='source'):layout.placement()
    value['so101_support']['measurement_source']='test fixture, not a real measurement'
    path.write_text(json.dumps(value))
    result=layout.placement()
    assert result['effective_so101_height_m']==.8
    assert result['so101_height_status']=='measured_height_only'
    assert result['common_world_transform_verified'] is False


@pytest.mark.parametrize('height',[True,-1,0,2,'0.75'])
def test_invalid_height_rejected(config,height):
    path,value=config;value['so101_support']['nominal_simulation_height_m']=height
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):layout.placement()
