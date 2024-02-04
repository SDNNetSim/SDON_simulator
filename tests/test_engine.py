import unittest
import os
import json
from unittest.mock import MagicMock, patch

import numpy as np

from sim_scripts.engine import Engine


class TestEngine(unittest.TestCase):
    """
    Methods related to testing engine.py
    """

    def setUp(self):
        engine_props = {'ai_algorithm': None, 'network': 'USNet', 'date': '2024-02-02', 'sim_start': '00:00'}
        self.engine = Engine(engine_props=engine_props)
        self.engine.reqs_dict = {1.0: {'req_id': 10}}

        self.engine.sdn_obj = MagicMock()
        sdn_path = os.path.join('fixtures', 'sdn_props.json')
        with open(sdn_path, 'r', encoding='utf-8') as file_obj:
            self.engine.sdn_obj.sdn_props = json.load(file_obj)

        engine_path = os.path.join('fixtures', 'engine_props.json')
        with open(engine_path, 'r', encoding='utf-8') as file_obj:
            self.engine.engine_props = json.load(file_obj)

        self.engine.topology = MagicMock()
        self.engine.net_spec_dict = {}
        self.engine.reqs_status_dict = {}
        self.engine.ai_obj = MagicMock()
        self.engine.stats_obj = MagicMock()

    def test_handle_arrival(self):
        """
        Tests the handle arrival method.
        """
        curr_time = 1.0
        self.engine.handle_arrival(curr_time=curr_time)

        self.engine.sdn_obj.handle_event.assert_called_once_with(request_type='arrival')
        self.assertEqual(self.engine.net_spec_dict, {'updated': 'value'})
        self.engine.stats_obj.iter_update.assert_called_once_with(req_data={'req_id': 10},
                                                                  sdn_data=self.engine.sdn_obj.sdn_props)

        req_status = self.engine.reqs_status_dict[self.engine.reqs_dict[curr_time]['req_id']]
        self.assertEqual(req_status, {
            "mod_format": 'QAM',
            "path": ['A', 'B', 'C'],
            "is_sliced": False,
            "was_routed": True
        })

    def test_handle_release_with_req(self):
        """
        Test handle release with an existing request in the reqs_status_dict.
        """
        curr_time = 1.0
        req_id = self.engine.reqs_dict[curr_time]['req_id']
        self.engine.reqs_status_dict[req_id] = {
            'path': ['A', 'B', 'C']
        }
        self.engine.handle_release(curr_time=curr_time)

        self.engine.sdn_obj.handle_event.assert_called_once_with(request_type='release')
        self.assertEqual(self.engine.net_spec_dict, self.engine.sdn_obj.sdn_props['net_spec_dict'])

    def test_handle_release_without_req(self):
        """
        Test handle release without an existing request in the reqs_status_dict.
        """
        curr_time = 1.0
        self.engine.handle_release(curr_time=curr_time)
        self.engine.sdn_obj.handle_event.assert_not_called()

    @patch('sim_scripts.engine.nx.Graph')
    def test_create_topology(self, mock_graph):
        """
        Tests the create topology method.
        """
        self.engine.topology = mock_graph.return_value
        self.engine.engine_props['topology_info']['nodes'] = ['A', 'B']
        self.engine.engine_props['topology_info']['links'] = {
            1: {"source": "A", "destination": "B", "fiber": {"num_cores": 2}, "length": 100, "spectral_slots": 320}
        }
        self.engine.create_topology()

        self.engine.topology.add_nodes_from.assert_called_once_with(['A', 'B'])
        self.engine.topology.add_edge.assert_called_once_with('A', 'B', length=100, nli_cost=None)
        expected_cores_matrix = np.zeros((2, 320))
        expected_net_spec = {
            ('A', 'B'): {'cores_matrix': expected_cores_matrix, 'link_num': 1},
            ('B', 'A'): {'cores_matrix': expected_cores_matrix, 'link_num': 1}
        }
        for link, link_data in self.engine.net_spec_dict.items():
            self.assertTrue(np.array_equal(link_data['cores_matrix'], expected_net_spec[link]['cores_matrix']))
            self.assertEqual(link_data['link_num'], expected_net_spec[link]['link_num'])

        self.assertEqual(self.engine.engine_props['topology'], self.engine.topology)
        self.assertEqual(self.engine.stats_obj.topology, self.engine.topology)
        self.assertEqual(self.engine.sdn_obj.sdn_props['net_spec_dict'], self.engine.net_spec_dict)
        self.assertEqual(self.engine.sdn_obj.sdn_props['topology'], self.engine.topology)


if __name__ == '__main__':
    unittest.main()
