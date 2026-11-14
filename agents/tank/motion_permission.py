"""Legacy diagnostic compatibility and unconditional robot-local velocity limits."""
import json
import time
from pathlib import Path

# Native app_cmd_vel_callback limits, requested by the operator after the
# reduced-speed bench check failed to turn the tracks.
BENCH_V = .2
BENCH_W = .5


def bench_authorized(root):
    try:
        permit = json.loads((root / 'bench-authorized.json').read_text())
        return (permit.get('scope') == 'suspended_tracks_only'
                and permit.get('operator_confirmed') is True
                and permit['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and permit['created_at'] <= time.time() < permit['expires_at']
                and 0 < permit['expires_at'] - permit['created_at'] <= 900)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def require_motion_permission(root):
    if (root / 'incident-block.json').exists() and not bench_authorized(root) and not neutral_diagnostic_authorized(root):
        raise PermissionError('Motion blocked after STOP incident; confirmed bench permission required')


def neutral_diagnostic_authorized(root):
    try:
        permit=json.loads((root/'neutral-diagnostic.json').read_text())
        return (permit.get('scope')=='zero_commands_only'
                and permit['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and permit['created_at']<=time.time()<permit['expires_at']
                and 0<permit['expires_at']-permit['created_at']<=90)
    except (OSError,ValueError,KeyError,TypeError):return False


def limited_command(root, v, w, mode='manual'):
    if (root/'neutral-diagnostic.json').exists():
        if not neutral_diagnostic_authorized(root) or v!=0. or w!=0.:
            raise PermissionError('Neutral-only diagnostic: movement is forbidden')
        return 0.,0.
    require_motion_permission(root)
    if bench_authorized(root):
        return max(-BENCH_V, min(BENCH_V, v)), max(-BENCH_W, min(BENCH_W, w))
    if __package__:from .limits import load,clip
    else:from limits import load,clip
    return clip(load(root),v,w,mode)
