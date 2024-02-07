import unittest
from unittest.mock import MagicMock

import numpy as np

from helper_scripts.stats_helpers import SimStats


class TestSimStats(unittest.TestCase):

    def setUp(self):
        self.engine_props = {}
        self.sim_info = "Simulation Info"
        self.stats_props = {'snapshots_dict': {
            2: {'occupied_slots': [], 'guard_slots': [], 'active_requests': [], 'blocking_prob': [],
                'num_segments': []}}}
        self.sim_stats = SimStats(engine_props=self.engine_props, sim_info=self.sim_info, stats_props=self.stats_props)

    def test_get_snapshot_info(self):
        net_spec_dict = {
            (0, 1): {'cores_matrix': [np.array([0, 1, 0, -1]), np.array([1, 0, -1, 0])]},
            (1, 0): {'cores_matrix': [np.array([0, -1, 1, 0]), np.array([-1, 1, 0, 0])]}
        }
        path_list = [(0, 1)]
        occupied_slots, guard_slots, active_reqs = SimStats._get_snapshot_info(net_spec_dict=net_spec_dict,
                                                                               path_list=path_list)

        self.assertEqual(occupied_slots, 4)
        self.assertEqual(guard_slots, 2)
        self.assertEqual(active_reqs, 1)

    def test_update_snapshot(self):
        self.sim_stats._get_snapshot_info = MagicMock(return_value=(3, 3, 1))
        self.sim_stats.blocked_reqs = 1
        req_num = 2
        path_list = [(0, 1)]

        self.sim_stats.update_snapshot(net_spec_dict={}, req_num=req_num, path_list=path_list)

        self.assertIn(req_num, self.sim_stats.stats_props['snapshots_dict'])
        snapshot = self.sim_stats.stats_props['snapshots_dict'][req_num]
        self.assertEqual(snapshot['occupied_slots'][0], 3)
        self.assertEqual(snapshot['guard_slots'][0], 3)
        self.assertEqual(snapshot['active_requests'][0], 1)
        self.assertAlmostEqual(snapshot['blocking_prob'][0], 0.5)
        self.assertEqual(snapshot['num_segments'][0], self.sim_stats.curr_trans)
