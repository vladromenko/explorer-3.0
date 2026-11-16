"""Bound the actual training process tree, including children of the official trainer."""
import re
import subprocess
import sys


def unit_name(ident):
    if not re.fullmatch(r'[a-f0-9]{32}', ident):
        raise ValueError('Invalid job identifier')
    return f'explorer3-job-{ident}.scope'


def command(ident, args, config):
    if config.get('kind', 'train') not in {'train', 'sim_evaluate'}:
        return args, None
    # Homebrain uses a cgroup: limiting the queue's launcher does not limit the
    # separate UI process which actually starts a job. Never fall back uncapped.
    if sys.platform != 'linux':
        raise RuntimeError('Managed training requires the Homebrain Linux user service')
    memory = config.get('memory_limit_mib', 6144)
    if type(memory) is not int or not 1024 <= memory <= 8192:
        raise ValueError('Training memory limit must be 1024–8192 MiB')
    unit = unit_name(ident)
    return ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--unit', unit,
            '-p', f'MemoryMax={memory}M', '-p', 'MemorySwapMax=2048M',
            '-p', 'OOMPolicy=kill', *args], unit


def stop(unit):
    """Stop the cgroup, including grandchildren that started new process groups."""
    if unit is None:
        return
    if not re.fullmatch(r'explorer3-job-[a-f0-9]{32}\.scope', unit):
        raise ValueError('Invalid compute unit')
    try:
        subprocess.run(['systemctl', '--user', 'stop', unit], timeout=12,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', unit], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
