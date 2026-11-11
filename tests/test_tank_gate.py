import math
import time
import unittest
from agents.tank.agent import Hardware,targets

class TankGateTests(unittest.TestCase):
    def test_existing_chassis_mapping_and_limits(self):
        action,motors=targets(2.,0.)
        self.assertEqual(action,[.2,0.])
        expected=.2/(math.pi*.075)
        self.assertEqual(motors,[[1,-expected],[2,-expected],[3,expected],[4,expected]])
        self.assertEqual(targets(0.,10.)[0],[0.,.5])
        with self.assertRaises(ValueError):targets(float('nan'),0.)

    def test_timeout_stops_and_rejects_late_command(self):
        class FakeBoard:
            commands=[]
            def get_battery(self):return None
            def get_imu(self):return None
            def set_motor_speed(self,value):self.commands.append(value)
        h=Hardware()
        try:
            self.assertFalse(h.armed)
            self.assertIsNone(h.board)
            h.board=FakeBoard();h.armed=True;h.lease='old';h.expires=time.monotonic()-.1
            time.sleep(.08)
            self.assertFalse(h.armed)
            self.assertEqual(h.board.commands[-1],[[1,0.],[2,0.],[3,0.],[4,0.]])
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
