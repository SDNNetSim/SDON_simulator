# pylint: disable=protected-access

import unittest
from unittest.mock import patch

import numpy as np

from sim_scripts.sdn_controller import SDNController


class TestSDNController(unittest.TestCase):
    """
    Tests the SDNController class.
    """

    def setUp(self):
        self.engine_props = {'cores_per_link': 7, 'guard_slots': 1, 'route_method': 'shortest_path', 'max_segments': 1}
        self.controller = SDNController(engine_props=self.engine_props)

        self.controller.sdn_props = {
            'bandwidth_list': [],
            'bandwidth': '50',
            'stat_key_list': ['xt_cost', 'core_num'],
            'xt_cost': [],
            'core_num': [],
            'was_routed': True,
            'net_spec_dict': {
                ('A', 'B'): {'cores_matrix': np.zeros((7, 10))},
                ('B', 'A'): {'cores_matrix': np.zeros((7, 10))},
                ('B', 'C'): {'cores_matrix': np.zeros((7, 10))},
                ('C', 'B'): {'cores_matrix': np.zeros((7, 10))},
            },
            'path_list': ['A', 'B', 'C'],
            'req_id': 1
        }

    def test_release(self):
        """
        Test the release method.
        """
        self.controller.sdn_props['net_spec_dict'][('A', 'B')]['cores_matrix'][0][:3] = 1
        self.controller.sdn_props['net_spec_dict'][('A', 'B')]['cores_matrix'][0][3] = -1
        self.controller.release()

        for link in zip(self.controller.sdn_props['path_list'], self.controller.sdn_props['path_list'][1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.controller.sdn_props['net_spec_dict'][link]['cores_matrix'][core_num]
                self.assertTrue(np.all(core_arr[:4] == 0), "Request and guard band not properly cleared")

    def test_allocate(self):
        """
        Test the allocate method.
        """
        self.controller.spectrum_obj.spectrum_props = {
            'start_slot': 0,
            'end_slot': 3,
            'core_num': 0
        }
        self.controller.engine_props['guard_slots'] = True
        self.controller.allocate()

        for link in zip(self.controller.sdn_props['path_list'], self.controller.sdn_props['path_list'][1:]):
            core_matrix = self.controller.sdn_props['net_spec_dict'][link]['cores_matrix'][0]
            self.assertTrue(np.all(core_matrix[:2] == self.controller.sdn_props['req_id']),
                            "Request not properly allocated")

            self.assertEqual(core_matrix[2], self.controller.sdn_props['req_id'] * -1,
                             msg="Guard band not properly allocated.")

    def test_update_req_stats(self):
        """
        Test the update request statistics method.
        """
        self.controller.sdn_props = {
            'bandwidth_list': [],
            'stat_key_list': ['xt_cost', 'core_num'],
            'xt_cost': [],
            'core_num': []
        }
        self.controller.spectrum_obj.spectrum_props = {'xt_cost': 10, 'core_num': 2}

        self.controller._update_req_stats(bandwidth='100G')

        # Verify that the bandwidth and other stats are updated correctly
        self.assertIn('100G', self.controller.sdn_props['bandwidth_list'])
        self.assertIn(10, self.controller.sdn_props['xt_cost'])
        self.assertIn(2, self.controller.sdn_props['core_num'])

    @patch('sim_scripts.sdn_controller.SDNController.allocate')
    @patch('sim_scripts.sdn_controller.SDNController._update_req_stats')
    @patch('sim_scripts.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_allocate_slicing(self, mock_get_spectrum, mock_update_req_stats, mock_allocate):
        """
        Tests the allocate slicing method.
        """
        self.controller.spectrum_obj.spectrum_props = {'is_free': True}
        self.controller._allocate_slicing(num_segments=2, mod_format='QPSK', path_list=['A', 'B'], bandwidth='50G')

        mock_get_spectrum.assert_called()
        mock_allocate.assert_called()
        mock_update_req_stats.assert_called_with(bandwidth='50G')
        self.assertTrue(self.controller.sdn_props['was_routed'])

    @patch('sim_scripts.sdn_controller.SDNController.release')
    def test_handle_event_departure(self, mock_release):
        """
        Tests handle event with a departure request.
        """
        self.controller.handle_event(request_type="release")
        mock_release.assert_called_once()

    @patch('sim_scripts.sdn_controller.SDNController.allocate')
    @patch('sim_scripts.sdn_controller.SDNController._update_req_stats')
    @patch('sim_scripts.routing.Routing.get_route')
    @patch('sim_scripts.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_handle_event_arrival(self, mock_allocate, mock_stats, mock_route, mock_spectrum):
        """
        Tests the handle event with an arrival request.
        """
        mock_route.return_value = None
        self.controller.route_obj.route_props = {'paths_list': [['A', 'B', 'C']], 'mod_formats_list': [['QPSK']],
                                                 'weights_list': [10]}

        self.controller.spectrum_obj.spectrum_props = {'is_free': True}
        mock_spectrum.return_value = None

        mock_stats.return_value = None
        self.controller.handle_event(request_type="arrival")

        mock_allocate.assert_called_once()
        self.assertTrue(self.controller.sdn_props['was_routed'])
