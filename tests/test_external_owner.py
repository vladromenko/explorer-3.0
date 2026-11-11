import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from explorer3.control import Control
from explorer3.hardware import HardwareSource


class ExternalOwnerTests(unittest.TestCase):
    def test_reservation_blocks_even_previously_authorized_start_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'configs/so101-external-owner.json').write_text('{}')
            (root/'stage-b-authorized.json').write_text('{}')
            with patch('explorer3.control.DATA', root):
                self.assertFalse(Control().authorized())
            with patch('explorer3.hardware.DATA', root), patch('explorer3.hardware.ResourceLease') as lease, patch('explorer3.hardware.calibration_snapshot', return_value={}):
                with self.assertRaises(PermissionError):
                    HardwareSource().start(explicit_user_start=True)
                lease.assert_not_called()
