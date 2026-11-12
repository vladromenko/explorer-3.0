#!/usr/bin/env python3
"""Offline report from a saved native-controller capture; never connects to a robot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.rosmaster.feedback_diagnostics import analyze_capture, explanation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze_capture(json.loads(args.capture.read_text()))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(explanation(report))


if __name__ == '__main__':
    main()
