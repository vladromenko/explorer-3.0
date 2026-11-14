import math
import time
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from agents.tank.agent import Hardware
from agents.tank.serial_guard import CheckedSerial

class TankGateTests(unittest.TestCase):
    def test_control_uses_the_single_fleet_admission(self):
        from explorer3.control import Control
        control=Control();control.robot='tank'
        with patch('explorer3.control.local_work.hardware_allowed',return_value=False):
            self.assertFalse(control.authorized())
        with patch('explorer3.control.local_work.hardware_allowed',return_value=True):
            self.assertTrue(control.authorized())

    def test_neutral_diagnostic_never_grants_movement_even_without_incident(self):
        from agents.tank.motion_permission import limited_command,require_motion_permission
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            permit={'scope':'zero_commands_only','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'created_at':time.time()-1,'expires_at':time.time()+60}
            (root/'neutral-diagnostic.json').write_text(json.dumps(permit))
            require_motion_permission(root)
            self.assertEqual(limited_command(root,0.,0.),(0.,0.))
            for action in ([.01,0.],[0.,-.01]):
                with self.assertRaises(PermissionError):limited_command(root,*action)
            (root/'incident-block.json').write_text('{}')
            self.assertEqual(limited_command(root,0.,0.),(0.,0.))
            with patch('agents.tank.motion_permission.time.time',return_value=permit['expires_at']+1):
                with self.assertRaises(PermissionError):limited_command(root,0.,0.)

    def test_serial_loss_latches_incident_and_never_claims_a_neutral_write(self):
        class BrokenBoard:
            process=None
            def command(self,*args):raise OSError(5,'Input/output error')
            def stop(self):raise OSError(5,'Input/output error')
            def close(self):raise RuntimeError('Neutral write failed during close')
        with tempfile.TemporaryDirectory() as directory,patch('agents.tank.agent.ROOT',Path(directory)):
            h=Hardware();h.done.set();h.watchdog.join(timeout=1)
            h.board=BrokenBoard();h.armed=True;h.lease='live';h.expires=time.monotonic()+10
            h.last_action=[.2,-.5]
            with self.assertRaisesRegex(RuntimeError,'STOP не доставлен'):h.command('live',1,.2,0.)
            self.assertFalse(h.armed)
            self.assertFalse(h.neutral_written)
            self.assertEqual(h.last_action,[.2,-.5])
            self.assertFalse(h.status()['physical_stop_confirmed'])
            self.assertTrue((Path(directory)/'incident-block.json').exists())
            with self.assertRaises(PermissionError):h.command('live',2,0.,0.)
            with self.assertRaises(RuntimeError):h.close()
            self.assertIsNone(h.board)
            with self.assertRaises(RuntimeError):h.start()

    def test_bench_permission_expires_and_keeps_incident_block(self):
        from agents.tank.motion_permission import bench_authorized,limited_command
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'incident-block.json').write_text('{}')
            with self.assertRaises(PermissionError):limited_command(root,.1,.3)
            permit={'scope':'suspended_tracks_only','operator_confirmed':True,
                    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'created_at':time.time()-1,'expires_at':time.time()+60}
            (root/'bench-authorized.json').write_text(json.dumps(permit))
            self.assertTrue(bench_authorized(root))
            self.assertEqual(limited_command(root,.8,-.9),(.2,-.5))
            self.assertEqual(limited_command(root,.1,-.3),(.1,-.3))
            with patch('agents.tank.motion_permission.time.time',return_value=permit['expires_at']+1):
                with self.assertRaises(PermissionError):limited_command(root,.1,0.)
            permit['boot_id']='previous-boot'
            (root/'bench-authorized.json').write_text(json.dumps(permit))
            self.assertFalse(bench_authorized(root))
            self.assertTrue((root/'incident-block.json').exists())

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

    def test_camera_gap_stops_motion_and_requires_neutral_before_manual_resume(self):
        class Board:
            def __init__(self):self.sent=[]
            def status(self):return {'battery_voltage':8.,'battery_received_ns':time.monotonic_ns(),'imu':None,'imu_received_ns':None,'latched':False}
            def command(self,v,w):self.sent.append([v,w]);return {'action':[v,w]}
            def stop(self):self.sent.append('STOP')
        for mode in ('manual','policy'):
            with self.subTest(mode=mode):
                h=Hardware()
                try:
                    h.board=Board();h.mode=mode;h.armed=True;h.lease='operator';h.expires=time.monotonic()+3
                    h.image_info={'received_ns':time.monotonic_ns()-600_000_000}
                    deadline=time.monotonic()+1
                    while not h.board.sent and time.monotonic()<deadline:time.sleep(.01)
                    if mode=='policy':
                        self.assertFalse(h.armed);self.assertEqual(h.board.sent,['STOP'])
                    else:
                        self.assertTrue(h.armed);self.assertTrue(h.motion_paused)
                        self.assertEqual(h.board.sent,[[0.,0.]])
                        h.image_info={'received_ns':time.monotonic_ns()}
                        self.assertEqual(h.command('operator',1,.2,.5)['action'],[0.,0.])
                        self.assertTrue(h.motion_paused)
                        h.command('operator',2,0.,0.);self.assertFalse(h.motion_paused)
                        self.assertEqual(h.command('operator',3,.2,.5)['action'],[.2,.5])
                finally:h.done.set();h.watchdog.join(timeout=1)

    def test_key_release_switch_and_expiration_clear_motion(self):
        from unittest.mock import patch
        from explorer3.control import Control
        with patch('explorer3.control.catalog.event'),patch.object(Control,'authorized',return_value=True):
            control=Control();control.select('tank');control.acquire('operator')
            control.base_key('other','up',True)
            self.assertEqual(control.base_action(),[0.,0.])
            control.base_key('operator','up',True)
            self.assertEqual(control.base_action(),[.2,0.])
            control.base_key('operator','up',False)
            self.assertEqual(control.base_action(),[0.,0.])
            control.base_key('operator','left',True)
            control.select('so101')
            self.assertEqual(control.base_action(),[0.,0.])
            self.assertFalse(control.base_keys)
            control.select('tank');control.acquire('operator')
            control.base_key('operator','up',True);control.expires=time.monotonic()-.1
            self.assertEqual(control.base_action(),[0.,0.])

    def test_diagonals_and_keyboard_use_same_arcs_and_release_independently(self):
        from explorer3.control import Control
        with patch('explorer3.control.catalog.event'),patch.object(Control,'authorized',return_value=True):
            c=Control();c.select('tank');c.acquire('browser')
            for key,expected in [('up_left',[.2,.5]),('up_right',[.2,-.5]),
                                 ('down_left',[-.2,-.5]),('down_right',[-.2,.5])]:
                c.base_key('browser',key,True,'pointer:1');self.assertEqual(c.base_action(),expected)
                c.base_key('browser',key,False,'pointer:1');self.assertEqual(c.base_action(),[0.,0.])
            c.base_key('browser','down',True,'key:KeyS')
            c.base_key('browser','left',True,'key:KeyA')
            self.assertEqual(c.base_action(),[-.2,-.5])
            c.base_key('browser','down',True,'pointer:1')
            c.base_key('browser','down',False,'pointer:1')
            self.assertEqual(c.base_action(),[-.2,-.5])
            c.stop();self.assertFalse(c.base_inputs);self.assertEqual(c.base_action(),[0.,0.])

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
