"""Save N frames from the ROSMASTER arm camera (via Explorer :8030) for the E4 detector evaluation set.

  .venv/bin/python scripts/grab_frames.py empty_floor 3
  .venv/bin/python scripts/grab_frames.py distractors 3
  .venv/bin/python scripts/grab_frames.py all_three 3
Frames go to DATA/validation/e4_frames/<label>/<time>.jpg; one frame per ~1.5 s. Read-only: no motion.
"""
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer3.paths import DATA  # noqa: E402


def main():
    label, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = DATA / 'validation/e4_frames' / label
    out.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        with urllib.request.urlopen('http://127.0.0.1:8030/api/rosmaster/frame.jpg', timeout=5) as r:
            data = r.read()
        path = out / ('%s-%d.jpg' % (time.strftime('%Y%m%dT%H%M%S'), i))
        path.write_bytes(data)
        print(path, len(data))
        time.sleep(1.5)


if __name__ == '__main__':
    main()
