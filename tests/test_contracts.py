import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from explorer3.control import Control
from explorer3.training import split_episodes,train_command
from explorer3.resources import ResourceLease,ResourceBusy
from explorer3 import resources

class Contracts(unittest.TestCase):
    def test_startup_and_selection_disarm(self):
        with patch('explorer3.control.catalog.event'):
            c=Control()
            self.assertTrue(c.expired())
            with patch.object(c,'authorized',return_value=False):
                with self.assertRaises(PermissionError):
                    c.acquire('browser')
            with patch.object(c,'authorized',return_value=True):
                c.acquire('browser')
            c.select('tank')
            self.assertIsNone(c.owner)
            self.assertTrue(c.expired())
            self.assertEqual(c.mode,'Наблюдение')

    def test_groups_cannot_leak(self):
        episodes=[{'id':str(i),'session':str(i//2),'scene':str(i//2)} for i in range(20)]
        split=split_episodes(episodes)
        sets=[{(e['session'],e['scene']) for e in values} for values in split.values()]
        for i,a in enumerate(sets):
            for j,b in enumerate(sets):
                if i!=j:
                    self.assertFalse(a & b)
        self.assertEqual(sum(map(len,split.values())),20)

    def test_only_one_resource_owner(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(resources,'DATA',Path(directory)):
            with ResourceLease('compute'):
                with self.assertRaises(ResourceBusy):
                    with ResourceLease('compute'):
                        self.fail('Concurrent lease acquired')
            with ResourceLease('compute'):
                pass

    def test_trainer_is_official_without_shell(self):
        cmd=train_command(Path('/tmp/dataset'),Path('/tmp/checkpoint'))
        self.assertIn('lerobot.scripts.lerobot_train',cmd)
        self.assertIn('--policy.push_to_hub=false',cmd)
        self.assertIn('--steps=50',cmd)
        self.assertIn('--dataset.use_imagenet_stats=false',cmd)

if __name__=='__main__':
    unittest.main()
