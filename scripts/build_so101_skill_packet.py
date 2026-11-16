"""Admit an explicitly reviewed real donor recording; never opens hardware."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.so101_packet_builder import build_real_packet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    parser.add_argument('--review', required=True, type=Path)
    args = parser.parse_args()
    item, packet = build_real_packet(args.episode, json.loads(args.review.read_text()))
    print(json.dumps({'packet_sha256': item['skill_packet_sha256'], 'domain': packet['donor']['domain'],
                      'donor_grasp_verified': True, 'recipient_verified': False,
                      'training_started': False, 'motor_commands_sent': 0}))
