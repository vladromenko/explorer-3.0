"""Read-only inventory and private interprocess leases; never stop foreign owners."""
import fcntl
import os
from pathlib import Path
from .paths import DATA

class ResourceBusy(RuntimeError):
    pass

def owners(paths: list[str]) -> list[dict]:
    targets={str(Path(p).resolve()) for p in paths}
    found=[]
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) != os.getpid():
            try:
                hits=[]
                for fd in (proc/'fd').iterdir():
                    try:
                        link=os.readlink(fd)
                        if link in targets:
                            hits.append(link)
                    except FileNotFoundError:
                        pass  # Descriptor closed during read-only inventory.
                if hits:
                    found.append({'pid':int(proc.name),'devices':sorted(set(hits))})
            except (PermissionError,FileNotFoundError,ProcessLookupError):
                pass  # Kernel/other-user descriptors are protected; TIOCEXCL also required.
    return found

def assert_hardware_available(paths: list[str]) -> None:
    missing=[p for p in paths if not Path(p).exists()]
    busy=owners(paths)
    if missing or busy:
        raise ResourceBusy(f'Нет устройств: {missing}; владельцы: {busy}')

class ResourceLease:
    def __init__(self, name: str, *, shared=False):
        if not name.replace('-','').replace('_','').isalnum():
            raise ValueError('Invalid lease name')
        self.name,self.file=name,None
        self.shared=shared

    def __enter__(self):
        directory=DATA/'locks'
        directory.mkdir(parents=True,exist_ok=True)
        self.file=(directory/f'{self.name}.lock').open('a+')
        try:
            fcntl.flock(self.file,(fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            self.file=None
            raise ResourceBusy(f'Ресурс занят: {self.name}') from exc
        if not self.shared:
            self.file.seek(0)
            self.file.truncate()
            self.file.write(str(os.getpid()))
            self.file.flush()
        return self

    def __exit__(self,*args):
        if self.file:
            fcntl.flock(self.file,fcntl.LOCK_UN)
            self.file.close()
            self.file=None

class ExistingTrainingLease:
    """Coordinate with an existing lab lock without modifying its file or process."""
    def __init__(self):
        self.file=None

    def __enter__(self):
        path=Path.home()/'Projects/so101_ai_learning/runtime/locks/gpu-heavy.lock'
        if path.exists():
            self.file=path.open('r')
            try:
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.file.close();self.file=None
                raise ResourceBusy('GPU занят обучением прежнего проекта; Explorer ожидает') from exc
        return self

    def __exit__(self,*args):
        if self.file:
            fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close();self.file=None
