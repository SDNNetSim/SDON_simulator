# pylint: disable=protected-access
# pylint: disable=unused-argument

import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np

from helper_scripts.stats_helpers import SimStats
from arg_scripts.stats_args import SNAP_KEYS_LIST


class TestSimStats(unittest.TestCase):
    """
    Tests stats_helpers.py
    """

    def setUp(self):
        self.engine_props = {}
        self.sim_info = "Simulation Info"
        self.stats_props = {'snapshots_dict': {
            2: {'occupied_slots': [], 'guard_slots': [], 'active_requests': [], 'blocking_prob': [],
                'num_segments': []}},
            'mods_used_dict': {},
            'weights_dict': {'50GHz': {'QPSK': []}},
            'block_bw_dict': {'50GHz': 0},
            'cores_dict': {},
            'block_reasons_dict': {'congestion': 0, 'distance': 0},
            'sim_block_list': [],
            'hops_list': [],
            'lengths_list': [],
            'route_times_list': [],
        }
        self.sim_stats = SimStats(engine_props=self.engine_props, sim_info=self.sim_info, stats_props=self.stats_props)

    def test_get_snapshot_info(self):
        """
        Test get snapshot info.
        """
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
        """
        Test update snapshot.
        """
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

    def test_init_snapshots(self):
        """
        Test init snapshots.
        """
        self.sim_stats.engine_props = {'num_requests': 100, 'snapshot_step': 20}
        self.sim_stats._init_snapshots()

        expected_req_nums = list(range(0, 101, 20))
        for req_num in expected_req_nums:
            self.assertIn(req_num, self.sim_stats.stats_props['snapshots_dict'])
            for key in SNAP_KEYS_LIST:
                self.assertIn(key, self.sim_stats.stats_props['snapshots_dict'][req_num])
                self.assertEqual(self.sim_stats.stats_props['snapshots_dict'][req_num][key], [])

    def test_init_mods_weights_bws(self):
        """
        Test initialize modulations, weights, and bandwidths.
        """
        self.sim_stats.engine_props = {
            'mod_per_bw': {
                '50GHz': {'QPSK': {}, '16QAM': {}},
                '75GHz': {'QPSK': {}}
            }
        }

        self.sim_stats._init_mods_weights_bws()
        for bandwidth, mod_obj in self.sim_stats.engine_props['mod_per_bw'].items():
            self.assertIn(bandwidth, self.sim_stats.stats_props['mods_used_dict'])
            self.assertIn(bandwidth, self.sim_stats.stats_props['weights_dict'])
            self.assertIn(bandwidth, self.sim_stats.stats_props['block_bw_dict'])

            for modulation in mod_obj.keys():
                self.assertIn(modulation, self.sim_stats.stats_props['mods_used_dict'][bandwidth])
                self.assertIn(modulation, self.sim_stats.stats_props['weights_dict'][bandwidth])
                self.assertEqual(self.sim_stats.stats_props['mods_used_dict'][bandwidth][modulation], 0)
                self.assertEqual(self.sim_stats.stats_props['weights_dict'][bandwidth][modulation], [])

    def test_init_stat_dicts(self):
        """
        Test init statistic dictionaries.
        """
        self.sim_stats.engine_props = {
            'cores_per_link': 4,
            'save_snapshots': True,
            'mod_per_bw': {'50GHz': {'QPSK': {}}}
        }
        self.sim_stats._init_mods_weights_bws = MagicMock()
        self.sim_stats._init_snapshots = MagicMock()
        self.sim_stats._init_stat_dicts()

        expected_cores_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        self.assertEqual(self.sim_stats.stats_props['cores_dict'], expected_cores_dict)

        expected_block_reasons_dict = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
        self.assertEqual(self.sim_stats.stats_props['block_reasons_dict'], expected_block_reasons_dict)

    def test_init_stat_lists(self):
        """
        Test init stat lists.
        """
        self.sim_stats.stats_props = {
            'sim_block_list': [1, 2, 3],
            'some_other_list': [4, 5, 6],
            'non_list_item': 'not a list'
        }
        self.sim_stats.iteration = 0
        self.sim_stats._init_stat_lists()

        self.assertEqual(self.sim_stats.stats_props['sim_block_list'], [])
        self.assertEqual(self.sim_stats.stats_props['some_other_list'], [])
        self.assertEqual(self.sim_stats.stats_props['non_list_item'], 'not a list')

        self.sim_stats.iteration = 1
        self.sim_stats.stats_props['sim_block_list'] = [1, 2, 3]
        self.sim_stats._init_stat_lists()
        self.assertEqual(self.sim_stats.stats_props['sim_block_list'], [1, 2, 3])

    def test_init_iter_stats(self):
        """
        Test init iter stats.
        """
        self.sim_stats._init_stat_dicts = MagicMock()
        self.sim_stats._init_stat_lists = MagicMock()

        self.sim_stats.init_iter_stats()
        self.sim_stats._init_stat_dicts.assert_called_once()
        self.sim_stats._init_stat_lists.assert_called_once()

        self.assertEqual(self.sim_stats.blocked_reqs, 0)
        self.assertEqual(self.sim_stats.total_trans, 0)

    def test_get_blocking(self):
        """
        Test get blocking.
        """
        self.sim_stats.engine_props = {'num_requests': 100}
        self.sim_stats.blocked_reqs = 20

        self.sim_stats.get_blocking()
        expected_blocking_prob = 0.2
        self.assertIn(expected_blocking_prob, self.sim_stats.stats_props['sim_block_list'])

    def test_handle_iter_lists(self):
        """
        Test handle iter lists.
        """
        sdn_data = {
            'stat_key_list': ['core_list', 'modulation_list', 'xt_list'],
            'core_list': [1, 2],
            'modulation_list': ['QPSK', '16QAM'],
            'bandwidth_list': ['50GHz', '75GHz'],
            'xt_list': [0.1, 0.2]
        }
        self.sim_stats.stats_props['cores_dict'] = {1: 0, 2: 0}
        self.sim_stats.stats_props['mods_used_dict'] = {'50GHz': {'QPSK': 0}, '75GHz': {'16QAM': 0}}
        self.sim_stats.stats_props['xt_list'] = []

        self.sim_stats._handle_iter_lists(sdn_data=sdn_data)
        self.assertEqual(self.sim_stats.stats_props['cores_dict'][1], 1)
        self.assertEqual(self.sim_stats.stats_props['cores_dict'][2], 1)
        self.assertEqual(self.sim_stats.stats_props['mods_used_dict']['50GHz']['QPSK'], 1)
        self.assertEqual(self.sim_stats.stats_props['mods_used_dict']['75GHz']['16QAM'], 1)

        self.assertEqual(self.sim_stats.stats_props['xt_list'], [0.1, 0.2])

    @patch('helper_scripts.stats_helpers.find_path_len', return_value=100)
    def test_iter_update(self, mock_path_len):
        """
        Test iter update.
        """
        req_data_blocked = {'bandwidth': '50GHz'}
        sdn_data_blocked = {'was_routed': False, 'block_reason': 'congestion'}
        self.sim_stats.iter_update(req_data=req_data_blocked, sdn_data=sdn_data_blocked)

        self.assertEqual(self.sim_stats.blocked_reqs, 1)
        self.assertEqual(self.sim_stats.stats_props['block_reasons_dict']['congestion'], 1)
        self.assertEqual(self.sim_stats.stats_props['block_bw_dict']['50GHz'], 1)

        req_data_routed = {'bandwidth': '50GHz'}
        sdn_data_routed = {
            'was_routed': True,
            'path_list': ['A', 'B', 'C'],
            'route_time': 10,
            'num_trans': 2,
            'path_weight': 5,
            'modulation_list': ['QPSK'],
            'bandwidth': '50GHz',
            'stat_key_list': [],
        }
        self.sim_stats.iter_update(req_data=req_data_routed, sdn_data=sdn_data_routed)

        self.assertEqual(self.sim_stats.stats_props['hops_list'][0], 2)
        self.assertEqual(self.sim_stats.stats_props['lengths_list'][0], 100)
        self.assertEqual(self.sim_stats.stats_props['route_times_list'][0], 10)
        self.assertEqual(self.sim_stats.total_trans, 2)
        self.assertEqual(self.sim_stats.stats_props['weights_dict']['50GHz']['QPSK'][0], 5)

    def test_get_iter_means(self):
        """
        Test get iter means.
        """
        self.sim_stats.stats_props['snapshots_dict'] = {1: {'occupied_slots': [10, 20, 30]}}
        self.sim_stats.stats_props['weights_dict'] = {'50GHz': {'QPSK': [1, 2, 3, 4]}}

        self.sim_stats._get_iter_means()
        expected_mod_obj = {'mean': 2.5, 'std': 1.2909944487358056, 'min': 1, 'max': 4}
        self.assertEqual(self.sim_stats.stats_props['weights_dict']['50GHz']['QPSK'], expected_mod_obj)
        self.assertEqual(self.sim_stats.stats_props['snapshots_dict'][1]['occupied_slots'], 20)

    def test_end_iter_update(self):
        """
        Test end iter update.
        """
        self.sim_stats.engine_props = {'num_requests': 100}
        self.sim_stats.blocked_reqs = 20
        self.sim_stats.total_trans = 300
        self.sim_stats.stats_props = {
            'block_reasons_dict': {'congestion': 15, 'distance': 5},
            'snapshots_dict': {},
            'trans_list': []
        }

        with patch.object(self.sim_stats, '_get_iter_means'):
            self.sim_stats.end_iter_update()

        expected_trans_mean = 300 / 80  # total_trans / (num_requests - blocked_reqs)
        self.assertIn(expected_trans_mean, self.sim_stats.stats_props['trans_list'])
        self.assertEqual(self.sim_stats.stats_props['block_reasons_dict']['congestion'], 15 / 20)
        self.assertEqual(self.sim_stats.stats_props['block_reasons_dict']['distance'], 5 / 20)

    def test_get_conf_inter(self):
        """
        Test get confidence interval.
        """
        self.sim_stats.stats_props = {
            'sim_block_list': [0.1, 0.2, 0.15, 0.25]
        }

        should_end = self.sim_stats.get_conf_inter()
        self.assertIsNotNone(self.sim_stats.block_mean)
        self.assertIsNotNone(self.sim_stats.block_ci)
        self.assertIsNotNone(self.sim_stats.block_ci_percent)
        self.assertFalse(should_end)

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.join", return_value='mocked/path/to/simulation_results')  # Adjusted mocked path
    @patch("helper_scripts.os_helpers.create_dir")
    def test_save_stats_json(self, mock_create_dir, mock_path_join, mock_file):
        """
        Test save stats with a json file.
        """
        self.sim_stats = SimStats(sim_info='', engine_props={})
        self.sim_stats.engine_props = {
            'file_type': 'json',
            'erlang': 10,
            'thread_num': 'thread_1'
        }
        self.sim_stats.sim_info = 'sim_test'
        self.sim_stats.block_mean = 0.2
        self.sim_stats.block_variance = 0.02
        self.sim_stats.block_ci = 0.05
        self.sim_stats.block_ci_percent = 5
        self.sim_stats.iteration = 1
        self.sim_stats.stats_props = {
            'trans_list': [3.5],
        }

        self.sim_stats.save_stats()
        mock_file.assert_called_once_with('mocked/path/to/simulation_results/10_erlang.json', 'w', encoding='utf-8')

        # Check if JSON data written to file is as expected
