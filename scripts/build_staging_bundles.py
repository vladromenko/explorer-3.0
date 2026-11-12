#!/usr/bin/env python3
"""Build local staging archives. Does not connect, install, or start services."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

REPO = Path(__file__).resolve().parents[1]
TANK_FILES = (
    'agent.py', 'native_client.py', 'native_driver.py', 'serial_guard.py',
    'handoff.py', 'stop_agent.py', 'emergency_stop.py', 'start-agent.sh', 'start-native.sh',
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError(f'Invalid archive path: {name}')


def archive(destination, files, platform):
    for name in files:
        safe_name(name)
    manifest = {
        'platform': platform, 'purpose': 'staging_only', 'physical_acceptance': False,
        'installs_or_starts_services': False, 'contains_credentials': False,
        'files': {name: digest(data) for name, data in sorted(files.items())},
    }
    payload = {**files, 'MANIFEST.json': encoded(manifest)}
    # Stable bytes permit verification of the exact archive transferred later.
    with tarfile.open(destination, 'w') as output:
        for name, data in sorted(payload.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            entry.mtime = 0
            output.addfile(entry, io.BytesIO(data))
    verify(destination)
    return {'archive': destination.name, 'sha256': digest(destination.read_bytes()),
            'files': len(files), 'platform': platform, 'physical_acceptance': False}


def verify(path):
    """Read and check an archive without extracting or executing anything."""
    with tarfile.open(path, 'r') as source:
        files = {}
        for entry in source.getmembers():
            safe_name(entry.name)
            if not entry.isfile() or entry.name in files:
                raise ValueError(f'Non-file or duplicate entry: {entry.name}')
            files[entry.name] = source.extractfile(entry).read()
    manifest = json.loads(files.pop('MANIFEST.json'))
    actual = {name: digest(data) for name, data in files.items()}
    if actual != manifest['files']:
        raise ValueError('Archive content does not match its manifest')
    return manifest


def build(runtime, output):
    output.mkdir(parents=True, exist_ok=True)
    native_root = runtime / 'provenance/tank-native-chain'
    native_manifest = json.loads((native_root / 'native-manifest.json').read_text())
    tank = {f'payload/{name}': (REPO / 'agents/tank' / name).read_bytes() for name in TANK_FILES}
    for name, record in native_manifest.items():
        safe_name(name)
        if not name.startswith('native/'):
            raise ValueError('Unexpected native source path')
        data = (native_root / name).read_bytes()
        if digest(data) != record.get('copy_sha256', record['source_sha256']):
            raise ValueError(f'Native copy changed: {name}')
        tank['payload/' + name] = data
    tank['payload/incident-block.json'] = encoded({
        'motion_blocked': True, 'reason': 'Runaway incident; physical STOP bench retest pending',
    })
    tank['provenance/native-manifest.json'] = encoded(native_manifest)
    tank['requirements.lock.txt'] = (REPO / 'configs/tank-agent-requirements.lock.txt').read_bytes()
    tank['software-validation.json'] = (REPO / 'docs/tank-native-software-validation.json').read_bytes()
    tank['READ-ME.txt'] = (
        'Tank: локальный пакет для будущего обновления собственного /home/ubuntu/explorer3.\n'
        'Это не установщик. Сейчас не копировать поверх работающего процесса и не запускать.\n'
        'payload содержит выбранную штатную цепочку и агент Explorer, без изменения алгоритмов производителя.\n'
        'Зависимости среды: существующий MentorPi / ROS Humble, ros_robot_controller_msgs, '
        'собственная venv и ранее установленный vendor/camera. Бинарники камеры сюда не входят.\n'
        'Секреты, конфигурация сети, состояние владения и разрешения на движение не включены.\n'
        'При будущей установке сохранить существующие секреты/камеру и incident-block.json; '
        'не переносить из архива разрешение на движение.\n'
        'До физической проверки STOP на подставке движение остаётся заблокировано.\n'
        'Программные проверки из software-validation.json не подтверждают остановку моторов.\n'
    ).encode()
    ros = {name: (REPO / name).read_bytes() for name in (
        'agents/rosmaster/feedback_diagnostics.py', 'scripts/analyze_rosmaster_capture.py',
    )}
    ros['reference/rollback-working-snapshot.json'] = (
        runtime / 'validation/rosmaster-rollback-working-snapshot.json').read_bytes()
    ros['READ-ME.txt'] = (
        'ROSMASTER: только анализ ранее записанной телеметрии, без ROS и serial. Python 3.\n'
        'Из корня распакованного пакета: python3 scripts/analyze_rosmaster_capture.py CAPTURE.json\n'
        'CAPTURE.json — результат существующего декодера; анализатор не собирает данные с робота.\n'
        'reference содержит хеши последнего проверенного отката, а не проверку текущего состояния.\n'
        'Пакет не меняет Console8090, MirrorMapper, знаки суставов, прошивку или автозапуск.\n'
        'Плавное приведение и отклонённое исправление feedback в пакет не включены.\n'
        'Причина raw32767/незавершённой остановки, а также влияние разряда пока не установлены.\n'
    ).encode()
    result = [archive(output / 'tank-staging.tar', tank, 'tank'),
              archive(output / 'rosmaster-diagnostics-staging.tar', ros, 'rosmaster')]
    (output / 'bundles.json').write_bytes(encoded(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(encoded(build(args.runtime, args.output)).decode())
