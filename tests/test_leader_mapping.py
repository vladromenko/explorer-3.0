import unittest
from agents.rosmaster.leader_mapping import LeaderMapping


class LeaderMappingTests(unittest.TestCase):
    def test_engagement_release_and_new_anchor_never_jump(self):
        mapper = LeaderMapping()
        robot = [90.]*6
        first = [-10., -100., 96., 77., -93., 27.]
        self.assertEqual(mapper.update(first, robot, 1., clutch=True, sample_age=.01), [0.]*6)
        moved = [first[0]+2, *first[1:]]
        velocity = mapper.update(moved, robot, 1.04, clutch=True, sample_age=.01)
        self.assertGreater(velocity[0], 0)
        self.assertEqual(velocity[1:], [0.]*5)
        self.assertEqual(mapper.update(moved, robot, 1.08, clutch=False, sample_age=.01), [0.]*6)
        self.assertEqual(mapper.update(moved, [100.]*6, 1.12, clutch=True, sample_age=.01), [0.]*6)

    def test_bad_packet_resets_anchor_and_limits_bound_velocity(self):
        mapper = LeaderMapping()
        mapper.update([0.]*6, [90.]*6, 1., clutch=True, sample_age=0)
        with self.assertRaises(ValueError):mapper.update([0.]*6, [90.]*6, 1.2, clutch=True, sample_age=.2)
        self.assertIsNone(mapper.anchor)
        mapper.update([0.]*6, [90.]*6, 2., clutch=True, sample_age=0)
        velocity = mapper.update([10.]*6, [90.]*6, 2.05, clutch=True, sample_age=0)
        self.assertTrue(all(abs(v) <= 12 for v in velocity))
