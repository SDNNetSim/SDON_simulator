import unittest
from unittest.mock import patch
import numpy as np
import networkx as nx

from src.routing import Routing


class TestRouting(unittest.TestCase):
    """
    Test methods in routing.py
    """

    def setUp(self):
        self.engine_props = {
            'topology': nx.Graph()
        }
        self.engine_props['topology'].add_edge('A', 'B', weight=1, xt_cost=10, length=1)
        self.engine_props['topology'].add_edge('B', 'C', weight=1, xt_cost=5, length=1)
        self.engine_props['topology'].add_edge('A', 'C', weight=3, xt_cost=100, length=2)
        self.sdn_props = {
            'net_spec_dict': {
                ('A', 'B'): {'cores_matrix': np.zeros((1, 10))},
                ('B', 'C'): {'cores_matrix': np.ones((1, 10))},
                ('A', 'C'): {'cores_matrix': np.zeros((1, 10))}
            },
            'source': 'A',
            'destination': 'C',
            'topology': self.engine_props['topology'],
            'mod_formats': {
                'QPSK': {'max_length': 10},
                '16-QAM': {'max_length': 20},
                '64-QAM': {'max_length': 30}
            }
        }
        self.route_props = {
            'mod_formats_list': [],
            'weights_list': [],
            'paths_list': []
        }
        self.instance = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.instance.sdn_props = self.sdn_props
        self.instance.route_props = self.route_props

    def test_find_most_cong_link(self):
        """
        Test the find most congested link method.
        """
        path_list = ['A', 'B', 'C']
        self.instance._find_most_cong_link(path_list)  # pylint: disable=protected-access
        self.assertEqual(len(self.instance.route_props['paths_list'][0]), 2)

        cores_arr = self.instance.route_props['paths_list'][0]['link_dict']['link']['cores_matrix'][0]
        condition = np.all(cores_arr == 1)
        self.assertTrue(condition)

    def test_find_least_cong(self):
        """
        Test the find the least congested route method.
        """
        self.instance.route_props['paths_list'] = [
            {'path_list': ['A', 'B', 'C'],
             'link_dict': {'link': self.sdn_props['net_spec_dict'][('A', 'B')], 'free_slots': 5}},
            {'path_list': ['A', 'C'],
             'link_dict': {'link': self.sdn_props['net_spec_dict'][('A', 'C')], 'free_slots': 10}}
        ]
        self.instance._find_least_cong()  # pylint: disable=protected-access
        self.assertEqual(len(self.instance.route_props['paths_list']), 1)
        self.assertEqual(self.instance.route_props['paths_list'][0], ['A', 'C'])

    def test_find_least_cong_path(self):
        """
        Test find the least congested path.
        """
        self.instance.find_least_cong()
        self.assertEqual(len(self.instance.route_props['paths_list']), 2)
        self.assertEqual(self.instance.route_props['paths_list'][0]['path_list'], ['A', 'C'])

    def test_least_xt_cost_path_selection(self):
        """
        Test find least cross-talk method.
        """
        self.instance.find_least_weight('xt_cost')

        expected_path = ['A', 'B', 'C']
        selected_path = self.instance.route_props['paths_list'][0]
        self.assertEqual(selected_path, expected_path, f"Expected path {expected_path} but got {selected_path}")

    def test_least_weight_path_selection(self):
        """
        Test find the least weight path method.
        """
        self.instance.find_least_weight('weight')

        expected_path = ['A', 'B', 'C']
        selected_path = self.instance.route_props['paths_list'][0]
        self.assertEqual(selected_path, expected_path, f"Expected path {expected_path} but got {selected_path}")

    def test_find_k_shortest_paths(self):
        """
        Test find the k-shortest paths method.
        """
        self.engine_props['k_paths'] = 2
        self.sdn_props['bandwidth'] = '50'
        self.engine_props['mod_per_bw'] = {
            '50': {
                'QPSK': {'max_length': 10},
                '16-QAM': {'max_length': 20},
                '64-QAM': {'max_length': 20},
            }
        }
        self.instance.find_k_shortest()
        self.assertEqual(len(self.instance.route_props['paths_list']), self.engine_props['k_paths'],
                         "Did not find the expected number of shortest paths")
        for path in self.instance.route_props['paths_list']:
            self.assertIsInstance(path, list, "Each path should be a list")
        for mod_format_list in self.instance.route_props['mod_formats_list']:
            self.assertEqual(len(mod_format_list), 1, "Each path should have exactly one modulation format")
        for weight in self.instance.route_props['weights_list']:
            self.assertIsInstance(weight, (int, float), "Each weight should be a number")

    def test_find_least_nli(self):
        """
        Test find the least non-linear impairment cost method.
        """
        self.route_props['span_len'] = 80
        self.sdn_props['bandwidth'] = 'some_bandwidth'
        self.engine_props['mod_per_bw'] = {
            'some_bandwidth': {
                'QPSK': {'slots_needed': 10}
            }
        }
        with patch.object(self.instance.route_help_obj, 'get_nli_cost', return_value=1.0):
            self.instance.find_least_nli()

            for link_tuple in list(self.sdn_props['net_spec_dict'].keys())[::2]:
                source, destination = link_tuple
                self.assertIn('nli_cost', self.sdn_props['topology'][source][destination], "NLI cost not set for link")

    def test_find_least_xt(self):
        """
        Test find the least XT method.
        """
        self.route_props['span_len'] = 80
        self.engine_props['xt_type'] = 'with_length'
        self.engine_props['beta'] = 0.5
        self.route_props['max_span'] = 100
        self.route_props['max_link_length'] = 1000

        with patch('helper_scripts.sim_helpers.find_free_slots', return_value={'free_slots': []}), \
                patch.object(self.instance.route_help_obj, 'find_xt_link_cost', return_value=0.1), \
                patch.object(self.instance.route_help_obj, 'get_max_link_length', return_value=100):
            self.instance.find_least_xt()

            for link_list in list(self.sdn_props['net_spec_dict'].keys())[::2]:
                source, destination = link_list
                self.assertIn('xt_cost', self.sdn_props['topology'][source][destination], "XT cost not set for link")
