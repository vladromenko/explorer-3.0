"""Short, boot-bound bench permission; the unresolved incident marker stays intact."""
import json
import time
from pathlib import Path

BENCH_V = .03
BENCH_W = .12
BENCH_PULSE_S = .7


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
    if (root / 'incident-block.json').exists() and not bench_authorized(root):
        raise PermissionError('Motion blocked after STOP incident; confirmed bench permission required')


def limited_command(root, v, w):
    require_motion_permission(root)
    if bench_authorized(root):
        return max(-BENCH_V, min(BENCH_V, v)), max(-BENCH_W, min(BENCH_W, w))
    return v, w
