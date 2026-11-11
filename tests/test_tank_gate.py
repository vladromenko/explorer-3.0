import math
import time
import unittest
from agents.tank.agent import Hardware
from agents.tank.serial_guard import CheckedSerial

class TankGateTests(unittest.TestCase):
    def test_partial_serial_write_is_rejected(self):
        class Journal:
            records=[]
            def emit(self,value):self.records.append(value)
        class Port:
            out_waiting=0
            def write(self,packet):return len(packet)-1
        journal=Journal()
        port=CheckedSerial(Port(),journal)
        with self.assertRaises(IOError):port.write(bytes.fromhex('aa5503020501'))
        self.assertEqual(journal.records[-1]['event'],'serial_write_error')

    def test_timeout_stops_and_rejects_late_command(self):
        class FakeBoard:
            commands=[]
            def status(self):return {'battery_voltage':7.4,'battery_received_ns':time.monotonic_ns(),'imu':None,'imu_received_ns':None,'latched':False}
            def stop(self):self.commands.append('STOP')
        h=Hardware()
        try:
            self.assertFalse(h.armed)
            self.assertIsNone(h.board)
            h.board=FakeBoard();h.armed=True;h.lease='old';h.expires=time.monotonic()-.1
            time.sleep(.08)
            self.assertFalse(h.armed)
            self.assertEqual(h.board.commands[-1],'STOP')
            with self.assertRaises(PermissionError):h.command('old',1,.1,0.)
        finally:
            h.done.set();h.watchdog.join(timeout=1)

    def test_key_release_switch_and_expiration_clear_motion(self):
        from unittest.mock import patch
        from explorer3.control import Control
        with patch('explorer3.control.catalog.event'),patch.object(Control,'authorized',return_value=True):
            control=Control();control.select('tank');control.acquire('operator')
            control.base_key('other','up',True)
            self.assertEqual(control.base_action(),[0.,0.])
            control.base_key('operator','up',True)
            self.assertEqual(control.base_action(),[.1,0.])
            control.base_key('operator','up',False)
            self.assertEqual(control.base_action(),[0.,0.])
            control.base_key('operator','left',True)
            control.select('so101')
            self.assertEqual(control.base_action(),[0.,0.])
            self.assertFalse(control.base_keys)
            control.select('tank');control.acquire('operator')
            control.base_key('operator','up',True);control.expires=time.monotonic()-.1
            self.assertEqual(control.base_action(),[0.,0.])

    def test_incident_blocks_start_before_native_process(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import agents.tank.agent as agent
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'incident-block.json').write_text('{}')
            h=Hardware()
            try:
                with patch.object(agent,'ROOT',root),patch.object(agent,'NativeDriver') as driver:
                    with self.assertRaises(PermissionError):h.start()
                    driver.assert_not_called()
            finally:h.done.set();h.watchdog.join(timeout=1)

    def test_journal_failure_does_not_block_neutral(self):
        class Journal:
            def emit(self,value):return False
        class Port:
            out_waiting=0
            sent=[]
            def write(self,packet):self.sent.append(packet);return len(packet)
        raw=Port();port=CheckedSerial(raw,Journal())
        neutral=bytes.fromhex('aa5503160104000000000001000000000200000000030000000007')
        forward=bytes.fromhex('aa550316010400af4cd9be01af4cd9be02af4cd93e03af4cd93e98')
        self.assertEqual(port.write(neutral),27)
        with self.assertRaises(RuntimeError):port.write(forward)
        self.assertEqual(raw.sent,[neutral])
