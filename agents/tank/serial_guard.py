"""Checks the unchanged SDK's serial writes; OS delivery is never a motor acknowledgement."""
import json
import math
import queue
import struct
import threading
import time



class WriteJournal:
    """A bounded queue keeps filesystem I/O off the serial/watchdog path."""
    def __init__(self,path):
        self.pending=queue.Queue(maxsize=1000)
        self.closed=threading.Event()
        self.error=None
        self.path=path
        self.worker=threading.Thread(target=self._run,daemon=True)
        self.worker.start()

    def emit(self,entry):
        if self.error:return False
        try:self.pending.put_nowait({'monotonic_ns':time.monotonic_ns(),**entry})
        except queue.Full:
            self.error='Serial journal queue full';return False
        return True

    def _run(self):
        try:
            with self.path.open('a',buffering=1) as output:
                while not self.closed.is_set() or not self.pending.empty():
                    try:entry=self.pending.get(timeout=.1)
                    except queue.Empty:entry=None
                    if entry is not None:output.write(json.dumps(entry)+'\n')
        except Exception as exc:self.error=str(exc)

    def close(self):
        self.closed.set();self.worker.join(timeout=2)
        if self.worker.is_alive():raise RuntimeError('Serial journal did not close')
        if self.error:raise RuntimeError(self.error)


class CheckedSerial:
    def __init__(self,port,journal):
        self.port=port
        self.port.write_timeout=.1
        self.journal=journal

    def __getattr__(self,name):return getattr(self.port,name)

    def write(self,packet):
        # The SDK owns packet construction. Check only its emitted motor targets and write result.
        neutral=False
        if packet[:3]==b'\xaa\x55\x03' and packet[4]==1:
            if len(packet)!=27 or packet[3]!=22 or packet[5]!=4:
                raise ValueError('Unexpected four-motor packet')
            values=[struct.unpack_from('<Bf',packet,6+5*i) for i in range(4)]
            if [v[0] for v in values]!=[0,1,2,3] or not all(math.isfinite(v[1]) and abs(v[1])<=1.2 for v in values):
                raise ValueError('Invalid motor IDs or rotation targets')
            neutral=all(v[1]==0. for v in values)
        start=time.monotonic_ns()
        recorded=self.journal.emit({'event':'serial_write_attempt','packet_hex':packet.hex(),'motor_acknowledged':False})
        if recorded is False and not neutral:raise RuntimeError('Serial journal unavailable; only neutral is allowed')
        try:
            written=self.port.write(packet)
            if written!=len(packet):raise IOError(f'Partial serial write: {written}/{len(packet)}')
            deadline=time.monotonic()+.1
            while self.port.out_waiting and time.monotonic()<deadline:time.sleep(.001)
            if self.port.out_waiting:raise TimeoutError('Serial output queue did not drain')
        except Exception as exc:
            self.journal.emit({'event':'serial_write_error','error':str(exc),'packet_hex':packet.hex()})
            raise
        self.journal.emit({'event':'serial_write_complete','packet_hex':packet.hex(),'start_ns':start,
                           'bytes_written':written,'driver_queue_empty':True,'motor_acknowledged':False})
        return written
