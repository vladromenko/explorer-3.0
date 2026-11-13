"""One interactive operator; expired sessions never acquire control implicitly."""
import time


class OperatorLease:
    def __init__(self, timeout=2., clock=time.monotonic):
        self.timeout=timeout;self.clock=clock;self.owner=None;self.updated=0.

    def expire(self):
        expired=self.owner is not None and self.clock()-self.updated>self.timeout
        if expired:self.owner=None
        return expired

    def claim(self, token):
        self.expire()
        if not isinstance(token,str) or not 10<=len(token)<=100:
            raise ValueError('Недействительный сеанс страницы')
        if self.owner not in (None,token):
            raise ValueError('Сценой управляет другая страница. Нажмите STOP для передачи управления.')
        self.owner=token;self.updated=self.clock()

    def require(self, token):
        self.expire()
        if self.owner is None or token!=self.owner:
            raise ValueError('Управление остановлено. Нажмите «Двигать модель» или кнопку направления заново.')
        self.updated=self.clock()

    def release(self, token=None, *, force=False):
        if force or token==self.owner:self.owner=None
