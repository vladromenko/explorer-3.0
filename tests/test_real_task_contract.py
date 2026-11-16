"""Instruction contract regressions; no physical recordings are created."""
import copy

import pytest

from explorer3 import datasets, training_drafts
from explorer3.profiles import profile


def legacy_series(name, instruction):
    recorded=copy.deepcopy(profile(name))
    recorded.pop('task_template',None);recorded.pop('task_variant',None)
    recorded['task']=instruction
    entry={'id':'fixture','session':'series','profile':name}
    provenance={'domain':'real','profile':recorded,'hardware':{'collection':{
        'id':'series','profile':name,'instruction':instruction}}}
    return entry,provenance


@pytest.mark.parametrize('name,instruction',[
    ('rosmaster_pick_place','pick up the cube'),
    ('rosmaster_pick_place_rgbd_lidar','pick up the bottle'),
    ('so101_pick_place','hand over the cube')])
def test_recorded_current_instruction_can_match_without_rewriting_original(name,instruction):
    entry,recorded=legacy_series(name,instruction)
    before=copy.deepcopy(recorded)
    assert datasets.recording_contract(entry,recorded,profile(name)) == training_drafts.semantic_contract(profile(name))
    assert recorded==before


def test_catalogue_relabel_cannot_turn_box_or_donor_clip_into_handover():
    for instruction in ('Pick up the cube, place it in the box, and withdraw the arm.',
                        'Grasp the cube, lift it at least 5 cm, hold for 2 seconds, then put it back on the table.'):
        entry,recorded=legacy_series('so101_pick_place',instruction)
        with pytest.raises(ValueError,match='исходная серия'):
            datasets.recording_contract(entry,recorded,profile('so101_pick_place'))
    entry,recorded=legacy_series('rosmaster_pick_place','Pick up the cube, place it in the box, and withdraw the arm.')
    with pytest.raises(ValueError,match='исходная серия'):
        datasets.recording_contract(entry,recorded,profile('rosmaster_pick_place'))
    entry,recorded=legacy_series('rosmaster_pick_place','pick up the cube')
    recorded['hardware']['collection']['instruction']='put the cube in the box'
    with pytest.raises(ValueError,match='исходная серия'):
        datasets.recording_contract(entry,recorded,profile('rosmaster_pick_place'))


def test_instruction_normalization_does_not_remove_camera_or_units_checks():
    entry,recorded=legacy_series('rosmaster_pick_place','pick up the cube')
    recorded['profile']['camera']='another-camera'
    assert datasets.recording_contract(entry,recorded,profile('rosmaster_pick_place')) != training_drafts.semantic_contract(profile('rosmaster_pick_place'))
    old=copy.deepcopy(profile('rosmaster_pick_place'))
    old.pop('task_template');old.pop('task_variant');old['task']='Pick up the cube, place it in the box, and withdraw the arm.'
    assert training_drafts.semantic_contract(old) != training_drafts.semantic_contract(profile('rosmaster_pick_place'))
