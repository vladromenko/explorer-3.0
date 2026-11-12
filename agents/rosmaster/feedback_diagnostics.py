"""Analyze saved native-controller telemetry. No ROS, serial or actuator access.

Input is the output of the existing native decoder/read-only capture. A clean
snapshot is not a motion authorization or a physical acceptance result.
"""
from collections import Counter


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def analyze_capture(capture):
    rows = capture.get('servo', [])
    if not isinstance(rows, list):
        raise ValueError('servo must be a list of decoded telemetry records')
    joints = []
    for joint in range(1, 7):
        records = [r for r in rows if isinstance(r, dict) and r.get('joint') == joint]
        last = records[-1] if records else None
        raw = None if last is None else last.get('raw')
        valid = bool(last is not None and last.get('error') == 0 and last.get('device_error') == 0
                     and _integer(raw) and 96 <= raw <= 4000)
        joints.append({'joint': joint, 'samples': len(records), 'last_raw': raw,
                       'last_valid': valid, 'measurement_present': last is not None,
                       'error_counts': dict(Counter(str(r.get('error_name', r.get('error', 'unknown')))
                                                    for r in records))})
    status_rows = capture.get('status_tail', [])
    diag_rows = capture.get('arm_diagnostics', [])
    status = status_rows[-1] if status_rows else {}
    diag = diag_rows[-1] if diag_rows else {}
    pending = status.get('arm_cancel_pending')
    errors = []
    missing = [j['joint'] for j in joints if not j['measurement_present']]
    invalid = [j['joint'] for j in joints if j['measurement_present'] and not j['last_valid']]
    if missing:
        errors.append({'kind': 'missing_feedback', 'joints': missing})
    if invalid:
        errors.append({'kind': 'invalid_feedback', 'joints': invalid})
    if pending is True:
        errors.append({'kind': 'controller_cancel_pending'})
    if diag.get('write_error', 0):
        errors.append({'kind': 'controller_write_error', 'code': diag['write_error']})
    if capture.get('parser_errors', 0):
        errors.append({'kind': 'host_frame_parse_errors', 'count': capture['parser_errors']})
    elapsed = None
    now, expires = diag.get('acquired_us'), diag.get('expires_us')
    if _integer(now) and _integer(expires) and 0 < expires < now:
        elapsed = (now - expires) / 1e6
    return {
        'snapshot_only': True, 'analyzer_sends_commands': False, 'physical_acceptance': False,
        'joints': joints, 'anomalies': errors,
        'controller': {'arm_enabled': status.get('arm_enabled'), 'cancel_pending': pending,
                       'stop_reason': diag.get('stop_reason'), 'write_error': diag.get('write_error'),
                       'last_command_generation': diag.get('arm_generation'),
                       'last_sent_generation': diag.get('sent_generation'),
                       'seconds_since_arm_lease_expired': elapsed},
        'root_cause_established': False,
        'battery_causality': 'not established by these telemetry records',
        'motion_authorized': False,
    }


def explanation(report):
    invalid = [j for j in report['joints'] if j['measurement_present'] and not j['last_valid']]
    missing = [str(j['joint']) for j in report['joints'] if not j['measurement_present']]
    lines = ['Сохранённая телеметрия; состояние робота сейчас не проверялось.']
    if invalid:
        lines.append('Недопустимое измерение: ' + '; '.join(
            f"сустав {j['joint']}, raw={j['last_raw']} ({j['samples']} ответов)" for j in invalid) + '.')
    if missing:
        lines.append('Нет записанных измерений суставов: ' + ', '.join(missing) + '.')
    if report['controller']['cancel_pending']:
        lines.append('Контроллер сообщает о незавершённой остановке руки.')
    if report['controller']['write_error']:
        lines.append('Код ошибки записи контроллера: ' + str(report['controller']['write_error']) + '.')
    lines.append('Причина сбоя и влияние разряда не установлены; движение этим отчётом не разрешается.')
    return '\n'.join(lines)
