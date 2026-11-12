from agents.rosmaster.feedback_diagnostics import analyze_capture, explanation


def capture():
    return {'servo': [{'joint': j, 'raw': 2000 if j < 6 else 32767,
                       'error': 0 if j < 6 else 3, 'error_name': 'OK' if j < 6 else 'RANGE',
                       'device_error': 0} for j in range(1, 7)],
            'status_tail': [{'arm_enabled': False, 'arm_cancel_pending': True}],
            'arm_diagnostics': [{'acquired_us': 6000000, 'expires_us': 4000000,
                                 'write_error': 14, 'stop_reason': 14,
                                 'arm_generation': 12, 'sent_generation': 11}], 'parser_errors': 0}


def test_captured_failure_distinguishes_gripper_from_elbow_and_cause():
    report = analyze_capture(capture())
    assert report['joints'][2]['last_valid']
    assert not report['joints'][5]['last_valid']
    assert report['joints'][5]['measurement_present']
    assert report['controller']['cancel_pending']
    assert report['controller']['seconds_since_arm_lease_expired'] == 2
    assert not report['root_cause_established'] and not report['motion_authorized']
    assert 'сустав 6, raw=32767' in explanation(report)


def test_numeric_error_free_reply_outside_range_is_still_invalid():
    data = capture(); data['servo'][-1].update(error=0, error_name='OK')
    assert not analyze_capture(data)['joints'][-1]['last_valid']


def test_last_failed_measurement_is_not_replaced_by_prior_success():
    data = capture(); data['servo'].insert(0, dict(data['servo'][-1], raw=1300, error=0))
    assert not analyze_capture(data)['joints'][-1]['last_valid']


def test_no_data_never_becomes_synthetic_pose_or_ready():
    report = analyze_capture({})
    assert all(j['last_raw'] is None for j in report['joints'])
    assert not report['motion_authorized'] and report['snapshot_only']
    assert report['controller']['arm_enabled'] is None


def test_six_good_measurements_do_not_clear_controller_cancellation():
    data = capture(); data['servo'][-1].update(raw=1300, error=0, error_name='OK')
    report = analyze_capture(data)
    assert all(j['last_valid'] for j in report['joints'])
    assert {'kind': 'controller_cancel_pending'} in report['anomalies']
    assert not report['physical_acceptance']
