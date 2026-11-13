"""Explicit five-second USB microphone capture, no playback or motor access."""
import subprocess
import threading
from pathlib import Path

lock=threading.Lock()


def available():return Path('/proc/asound/Device/pcm0c').exists()


def capture():
    if not available():raise RuntimeError('USB capture device unavailable')
    if not lock.acquire(blocking=False):raise RuntimeError('Microphone capture already running')
    try:
        # WAV to stdout, exactly five seconds; no shell and no arbitrary device/args.
        result=subprocess.run(['arecord','-q','-D','plughw:CARD=Device,DEV=0','-t','wav',
                               '-f','S16_LE','-r','16000','-c','1','-d','5','-'],
                               capture_output=True,timeout=7)
        if result.returncode:raise RuntimeError('Microphone capture failed: '+result.stderr.decode(errors='replace')[:250])
        if not result.stdout.startswith(b'RIFF') or len(result.stdout)>200000:raise RuntimeError('Invalid audio capture')
        return result.stdout
    finally:lock.release()
