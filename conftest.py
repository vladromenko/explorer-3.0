"""Complete test inventory and explicit execution groups for recording readiness."""
from pathlib import Path
import pytest

SIM_FILES = {
    'test_sim_branching.py', 'test_fleet_execution.py', 'test_physical_layout.py',
    'test_rosmaster_sysid.py', 'test_sim_variation.py', 'test_sim_physics.py',
    'test_rosmaster_sim_contacts.py', 'test_sim_operator.py', 'test_sim_normalization.py',
    'test_sim_fleet.py', 'test_tank_camera_model.py',
}
# These existing tests actually fit an estimator or call a model. Keep visible,
# but do not execute under the operator's no-training/no-generation instruction.
DEFERRED = {
    'test_estimator_is_assignment_model_with_disjoint_evaluation',
    'test_real_act_uses_environment_input_without_training',
    'test_policy_and_frozen_normalization_roundtrip_without_training',
}

def test_group(path):
    name = Path(path).name
    return 'browser' if name.endswith('_browser.py') else ('sim' if name in SIM_FILES else 'main')

def pytest_configure(config):
    for group in ('main', 'sim', 'browser'):
        config.addinivalue_line('markers', f'{group}: recording readiness {group} group')


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(getattr(pytest.mark, test_group(item.path)))
        if item.originalname in DEFERRED:
            item.add_marker(pytest.mark.skip(reason='Operator prohibits training and model execution'))
