"""One simulation-only donor/recipient preflight, with raw captures and traceable lineage.

This checks the transfer mechanism, not a real SO-101 packet or learned skill.
Run in .venv-sim. Uses the existing EpisodeWriter via the official import path.
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMPLEMENTATION_SHA256 = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
    ('explorer3/sim/skill_transfer.py', 'explorer3/sim/aligned.py', 'scripts/skill_transfer_preflight.py')}
import numpy as np
from explorer3.paths import DATA, offline_environment
from explorer3.sim.capture import SimCapture
from explorer3.sim.factory import make, EXPERTS
from explorer3.sim.skill_transfer import body_pose, tcp_pose, so101_jaw_measurement, RetargetedGrasp, TransferVerifier, TransferRejected
from explorer3.skill_packets import relative_pose, admission
from explorer3.cognitive.store import digest


def ingest(root):
    subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'explorer3.sim.import_episode', str(root)],
                   cwd=ROOT, env=offline_environment(), check=True)
    return json.loads((root / 'imported.json').read_text())


def source_packet(seed, output):
    env = make('so101_pick_place_sim')
    phases = {name: {'name': name, 'reference': 'frozen_at_phase_start', 'samples': []}
              for name in ('approach', 'grasp', 'lift', 'hold')}
    trace, axes, held_time = [], [], 0.
    try:
        env.reset(seed=seed)
        expert = EXPERTS[env.profile](env)
        capture = SimCapture(env, seed, 'scripted_expert')
        while True:
            action = expert.action()
            capture.add(env, action)
            t = env.elapsed
            if 2. <= t < 8.:
                phase = 'approach' if t < 4. else 'grasp' if t < 6. else 'lift' if t < 7. else 'hold'
                entry = phases[phase]
                obj, tcp = body_pose(env.data, 'task_cube'), tcp_pose(env.data)
                if not entry['samples']:
                    entry['world_object_at_start'] = obj.tolist()
                jaw, axis = so101_jaw_measurement(env)
                entry['samples'].append({'time_s': t, 'world_tcp': tcp.tolist(),
                    'object_tcp': relative_pose(entry['world_object_at_start'], tcp).tolist(), 'jaw_opening_m': jaw})
                if phase == 'grasp':
                    axes.append(obj[:3, :3].T @ axis)
                lift = float(obj[2, 3] - .016)
                held = lift > .05 and np.linalg.norm(obj[:3, 3] - tcp[:3, 3]) < .06
                held_time = held_time + env.dt if held else 0.
                trace.append({'time_s': t, 'phase': phase, 'object_xyz': obj[:3, 3].tolist(),
                              'tcp_xyz': tcp[:3, 3].tolist(), 'lift_m': lift, 'held_s': held_time})
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        spool = capture.finish(info['outcome'], info['termination_reason'])
        imported = ingest(spool)
        report = {'domain': 'sim', 'method': 'simulation_ground_truth', 'trace': trace,
                  'whole_source_task': info['outcome'], 'source_spool': str(spool)}
        evidence = output / 'donor-verification.json'
        evidence.write_text(json.dumps(report, indent=2))
        best_hold = max(trace, key=lambda row: row['held_s'])
        good = info['outcome'] == 'success' and best_hold['held_s'] >= 1.
        axis = np.mean(axes[-5:], axis=0)
        axis /= np.linalg.norm(axis)
        packet = {'schema': 'explorer-skill-packet-v1', 'subskill': 'approach_grasp_lift_hold',
            'units': {'translation': 'm', 'time': 's', 'jaw_opening': 'm'},
            'frame_convention': 'T_a_b_maps_b_to_a;right_handed;phase_start_frozen',
            'donor': {'robot': 'so101', 'behavior_source': 'geometric', 'domain': 'sim',
                'episodes': [{'id': imported['episode_id'], 'sha256': imported['dataset_sha256']}],
                'usage': {'granted_by': 'owner', 'reference': 'software-only transfer preflight',
                          'allowed': ['research', 'adaptation', 'training']}},
            'object': {'instance_id': f'sim-so101-cube-{seed}', 'class': 'cube', 'surface': 'rigid_box',
                       'size_m': [.032] * 3, 'size_sigma_m': [0.] * 3},
            'phases': list(phases.values()),
            'measurement': {'pose_source': 'simulation_ground_truth', 'calibration_sha256': env.manifest['xml_sha256'],
                'camera_sha256': digest({'model': env.manifest, 'camera': env.camera}),
                'position_sigma_m': 0., 'orientation_sigma_rad': 0., 'jaw_sigma_m': 0.,
                'closing_axis_object': axis.tolist(),
                'jaw_definition': 'distance between distal fixed_jaw_box4 and moving_jaw_box2 collision pads; simulation geometry only'},
            'donor_verification': {'status': 'success' if good else 'failure',
                'lift_m': best_hold['lift_m'], 'held_s': best_hold['held_s'], 'method': 'simulation_ground_truth',
                'evidence_ref': str(evidence.relative_to(DATA)), 'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}}
        (output / 'packet.json').write_text(json.dumps(packet, indent=2))
        decision = admission(packet, domain='sim')
        if not decision['eligible']:
            raise ValueError('Donor simulation did not verify: ' + json.dumps(decision))
        return packet
    finally:
        env.close()


def recipient(packet, seed, output):
    env = make('rosmaster_floor_pick_sim')
    capture = None
    try:
        env.reset(seed=seed)
        expert = RetargetedGrasp(env, packet)
        verifier = TransferVerifier(env)
        env.physics_observer = verifier
        capture = SimCapture(env, seed, 'scripted_expert')
        capture.meta.update(transfer_lineage=expert.report, session='donor-' + digest(packet),
                            donor_packet_sha256=digest(packet))
        while True:
            try:
                action = expert.action()
            except TransferRejected as exc:
                info = {'outcome': 'failure', 'termination_reason': str(exc)}
                break
            capture.add(env, action)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        verification = verifier.report()
        if info['outcome'] == 'success' and not verification['accepted']:
            info.update(outcome='failure', termination_reason='independent_transfer_verification_failed')
        spool = capture.finish(info['outcome'], info['termination_reason'])
        imported = ingest(spool)
        report = {**expert.report, 'independent_verification': verification, 'recipient_seed': seed, 'outcome': info['outcome'],
                  'termination_reason': info['termination_reason'], 'recipient_import': imported,
                  'recipient_model': env.manifest, 'trace': expert.trace, 'recipient_spool': str(spool),
                  'learned_policy': False, 'physical_commands_sent': 0,
                  'implementation_sha256': IMPLEMENTATION_SHA256}
        (output / 'recipient.json').write_text(json.dumps(report, indent=2))
        return report
    except Exception as exc:
        if capture is not None:
            capture.finish('aborted', f'{type(exc).__name__}: {exc}')
        (output / 'recipient-error.json').write_text(json.dumps({'error': str(exc), 'type': type(exc).__name__}))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--donor-seed', type=int, default=41000)
    parser.add_argument('--recipient-seed', type=int, default=42000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acquisition-request', help='Link this attempt to an existing prepared request')
    parser.add_argument('--packet', type=Path, help='Reuse a verified donor artifact without collecting it again')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from explorer3.resources import ResourceLease
    with ResourceLease('compute'):
        packet = json.loads(args.packet.read_text()) if args.packet else source_packet(args.donor_seed, args.output)
        from explorer3.knowledge import add_packet
        add_packet(packet)
        report = recipient(packet, args.recipient_seed, args.output)
        if args.acquisition_request:
            from explorer3.cognitive.broker import record_transfer_attempt
            record_transfer_attempt(args.acquisition_request, args.output / 'recipient.json')
    print(json.dumps({'outcome': report['outcome'], 'packet_sha256': digest(packet), 'output': str(args.output)}), flush=True)
