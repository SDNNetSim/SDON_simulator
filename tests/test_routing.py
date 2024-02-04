import unittest
import numpy as np
import networkx as nx

from sim_scripts.routing import Routing


class TestRouting(unittest.TestCase):

    def setUp(self):
        self.engine_props = {
            'topology': nx.Graph()
        }
        self.engine_props['topology'].add_edge('A', 'B', weight=1)
        self.engine_props['topology'].add_edge('B', 'C', weight=1)
        self.engine_props['topology'].add_edge('A', 'C', weight=2)
        self.sdn_props = {
            'net_spec_dict': {
                ('A', 'B'): {'cores_matrix': np.zeros((1, 10))},
                ('B', 'C'): {'cores_matrix': np.ones((1, 10))},
                ('A', 'C'): {'cores_matrix': np.zeros((1, 10))}
            },
            'source': 'A',
            'destination': 'C'
        }

        self.routing = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)

    def test_find_most_cong_link(self):
        path_list = ['A', 'B', 'C']
        self.routing._find_most_cong_link(path_list)
        self.assertEqual(len(self.routing.route_props['paths_list'][0]['path_list']), 3)
        self.assertEqual(len(self.routing.route_props['paths_list']), 1)

        most_cong_link = self.routing.route_props['paths_list'][0]['link_dict']['link']
        cores_arr = most_cong_link['cores_matrix'][0]
        condition = np.all(cores_arr == 1)
        self.assertTrue(condition)

    def test_find_least_cong(self):
        self.routing.route_props['paths_list'] = [
            {'path_list': ['A', 'B', 'C'],
             'link_dict': {'link': self.sdn_props['net_spec_dict'][('A', 'B')], 'free_slots': 5}},
            {'path_list': ['A', 'C'],
             'link_dict': {'link': self.sdn_props['net_spec_dict'][('A', 'C')], 'free_slots': 10}}
        ]
        self.routing._find_least_cong()
        self.assertEqual(len(self.routing.route_props['paths_list']), 1)
        self.assertEqual(self.routing.route_props['paths_list'][0], ['A', 'C'])

    def test_find_least_cong_path(self):
        self.routing.find_least_cong()
        self.assertEqual(len(self.routing.route_props['paths_list']), 2)
        # Check that the least congested path is chosen
        self.assertIn(self.routing.route_props['paths_list'], (['A', 'B', 'C'], ['A', 'C']))


if __name__ == '__main__':
    unittest.main()
