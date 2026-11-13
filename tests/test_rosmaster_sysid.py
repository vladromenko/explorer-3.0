"""ROSMASTER response identification recovers known parameters from fixture recordings and refuses synthetic data as real."""
import pytest
from explorer3.sim import rosmaster_sysid as sysid

BASE = [{'delay_s': .08, 'tau_s': .15, 'gain': .8}, {'delay_s': .08, 'tau_s': .15, 'gain': .8}, {'delay_s': .1, 'tau_s': .25, 'gain': .9}]
ARM = [{'delay_s': .06, 'tau_s': .1, 'vmax_deg_s': 90.}] * 5


def entries(tmp_path, profile, params, count=4):
    return [{'id': f'e{i}', 'dataset': str(sysid.synthetic_episode(tmp_path / f'e{i}', profile, params, seed=i))} for i in range(count)]


def test_base_parameters_recovered_and_holdout_better_than_nominal(tmp_path):
    report = sysid.identify(entries(tmp_path, 'rosmaster_dock', BASE), 'rosmaster_dock', allow_synthetic=True)
    vx = report['channels']['vx_m_s']
    assert abs(vx['identified']['delay_s'] - .08) <= .04 and abs(vx['identified']['gain'] - .8) < .05
    assert .1 <= vx['identified']['tau_s'] <= .22
    assert vx['holdout_rmse']['identified'] < vx['holdout_rmse']['nominal_sim']
    assert report['status'] == 'candidate_not_deployed' and report['synthetic_input'] is True
    values = sysid.simulator_values(report)
    assert .1 < values['lag_s'] < .4 and abs(values['slip'] - .8) < .06


def test_arm_rate_limit_and_delay_recovered(tmp_path):
    report = sysid.identify(entries(tmp_path, 'rosmaster_pick_place', ARM, 3), 'rosmaster_pick_place', allow_synthetic=True)
    j = report['channels']['joint1']['identified']
    assert abs(j['delay_s'] - .06) <= .04 and j['vmax_deg_s'] in (60., 90., 120.)
    h = report['channels']['joint1']['holdout_rmse']
    assert h['identified'] < h['simulator_form'] <= h['nominal_sim'] * 1.01


def test_synthetic_refused_as_real(tmp_path):
    with pytest.raises(ValueError):
        sysid.identify(entries(tmp_path, 'rosmaster_dock', BASE, 2), 'rosmaster_dock')


def test_separate_servo_ages_are_not_collapsed_to_slowest():
    rows=[]
    for i in range(12):
        rows.append({'action_sent':[90]*5,'action_sent_agent_ns':int((i+.01)*1e9),
            'measured_response':[90]*5,'measured_response_age_s':[.01,.02,.03,.04,.05,.2],
            'agent_monotonic_ns':int(i*1e9),'controller_ready':True})
    _,_,mt,_=sysid.series(rows,'rosmaster_pick_place')
    assert mt[0].tolist()==pytest.approx([.99,.98,.97,.96,.95])
    rows[0]['measured_response_age_s'][0]=-1
    with pytest.raises(ValueError,match='age'):sysid.series(rows,'rosmaster_pick_place')


def test_same_scene_cannot_be_both_fit_and_holdout(tmp_path):
    data=entries(tmp_path,'rosmaster_dock',BASE,2)
    for e in data:e.update(session=e['id'],scene='same-scene')
    with pytest.raises(ValueError,match='независимые'):sysid.identify(data,'rosmaster_dock',allow_synthetic=True)
