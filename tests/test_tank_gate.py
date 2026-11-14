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



    def test_key_release_switch_and_expiration_clear_motion(self):
        from unittest.mock import patch
        from explorer3.control import Control
        with patch('explorer3.control.catalog.event'),patch.object(Control,'authorized',return_value=True):
            control=Control();control.select('tank');control.acquire('operator')
            control.base_key('other','up',True)
            self.assertEqual(control.base_action(),[0.,0.])
            control.base_key('operator','up',True)
            self.assertEqual(control.base_action(),[.3,0.])
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
            for key,expected in [('up_left',[.3,1.]),('up_right',[.3,-1.]),
                                 ('down_left',[-.3,-1.]),('down_right',[-.3,1.])]:
                c.base_key('browser',key,True,'pointer:1');self.assertEqual(c.base_action(),expected)
                c.base_key('browser',key,False,'pointer:1');self.assertEqual(c.base_action(),[0.,0.])
            c.base_key('browser','down',True,'key:KeyS')
            c.base_key('browser','left',True,'key:KeyA')
            self.assertEqual(c.base_action(),[-.3,-1.])
            c.base_key('browser','down',True,'pointer:1')
            c.base_key('browser','down',False,'pointer:1')
            self.assertEqual(c.base_action(),[-.3,-1.])
            c.stop();self.assertFalse(c.base_inputs);self.assertEqual(c.base_action(),[0.,0.])


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
