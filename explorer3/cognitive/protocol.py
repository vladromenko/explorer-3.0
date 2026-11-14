"""Shared decision conditions; this module never launches an experiment."""

VERSION='shared-helper-choices-v2'
CHOICES=('scan_then_dock','observe_then_dock','wait','human')
INSTRUCTION=(
    'Use the current observations to select a helper, wait, or request a human. '
    'scan_then_dock performs a bounded camera scan; observe_then_dock uses a fixed '
    'camera pose. Both are available with or without memory. Memory is fallible '
    'historical evidence, not a new observation or an execution permission. '
    'Neither plan may dock before NEW helper verification. '
    'Prefer the shorter observation only when the available evidence supports it; '
    'otherwise use a scan, wait, or human assistance. Do not infer success.')


def decision_request(comparison,observations,memory,model_config_sha256):
    if comparison not in ('M0','M1','M2','M3'):
        raise ValueError('Unknown comparison')
    return {'protocol_version':VERSION,'model_config_sha256':model_config_sha256,
        'scenario':'S1 fixed shared simulated dock','observations':observations,
        'allowed_choices':list(CHOICES),
        'memory':memory if comparison in ('M2','M3') else [],
        'instruction':INSTRUCTION}
