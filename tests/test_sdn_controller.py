# pylint: disable=protected-access

import unittest
from unittest.mock import patch
import numpy as np

from src.sdn_controller import SDNController
from arg_scripts.sdn_args import SDNProps  # Class import for sdn_props


class TestSDNController(unittest.TestCase):
    """
    Tests the SDNController class.
    """

    def setUp(self):
        self.engine_props = {
            'cores_per_link': 7,
            'guard_slots': 1,
            'route_method': 'shortest_path',
            'max_segments': 1,
            'band_list': ['c']
        }
        self.controller = SDNController(engine_props=self.engine_props)

        # Mock SDNProps to ensure it's treated as a class object
        self.controller.sdn_props = SDNProps()
        self.controller.sdn_props.path_list = ['A', 'B', 'C']
        self.controller.sdn_props.req_id = 1
        self.controller.sdn_props.net_spec_dict = {
            ('A', 'B'): {'cores_matrix': {'c': np.zeros((7, 10))}},
            ('B', 'A'): {'cores_matrix': {'c': np.zeros((7, 10))}},
            ('B', 'C'): {'cores_matrix': {'c': np.zeros((7, 10))}},
            ('C', 'B'): {'cores_matrix': {'c': np.zeros((7, 10))}}
        }

    def test_release(self):
        """
        Test the release method.
        """
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][:3] = 1
        self.controller.sdn_props.net_spec_dict[('A', 'B')]['cores_matrix']['c'][0][3] = -1
        self.controller.release()

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.controller.sdn_props.net_spec_dict[link]['cores_matrix']['c'][core_num]
                self.assertTrue(np.all(core_arr[:4] == 0), "Request and guard band not properly cleared")

    def test_allocate(self):
        """
        Test the allocate method.
        """
        self.controller.spectrum_obj.spectrum_props.start_slot = 0
        self.controller.spectrum_obj.spectrum_props.end_slot = 3
        self.controller.spectrum_obj.spectrum_props.core_num = 0
        self.controller.spectrum_obj.spectrum_props.curr_band = 'c'
        self.controller.engine_props['guard_slots'] = True
        self.controller.allocate()

        for link in zip(self.controller.sdn_props.path_list, self.controller.sdn_props.path_list[1:]):
            core_matrix = self.controller.sdn_props.net_spec_dict[link]['cores_matrix']['c'][0]
            self.assertTrue(np.all(core_matrix[:2] == self.controller.sdn_props.req_id),
                            "Request not properly allocated")

            self.assertEqual(core_matrix[2], self.controller.sdn_props.req_id * -1,
                             msg="Guard band not properly allocated.")

    def test_update_req_stats(self):
        """
        Test the update request statistics method.
        """
        # Properly initialize sdn_props with necessary attributes
        self.controller.sdn_props.xt_cost = []
        self.controller.sdn_props.core_num = []
        self.controller.spectrum_obj.spectrum_props.xt_cost = 10
        self.controller.spectrum_obj.spectrum_props.core_num = 2

        # Set sdn_props.stat_key_list to ensure the relevant keys are present
        self.controller.sdn_props.stat_key_list = ['xt_cost', 'core_num']

        # Call the method to update request statistics
        self.controller._update_req_stats(bandwidth='100G')

        # Verify that the bandwidth and other stats are updated correctly
        self.assertIn('100G', self.controller.sdn_props.bandwidth_list)
        self.assertIn(10, self.controller.sdn_props.xt_cost)  # Check if xt_cost was updated correctly
        self.assertIn(2, self.controller.sdn_props.core_num)  # Check if core_num was updated correctly

    @patch('src.sdn_controller.SDNController.allocate')
    @patch('src.sdn_controller.SDNController._update_req_stats')
    @patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_allocate_slicing(self, mock_get_spectrum, mock_update_req_stats, mock_allocate):
        """
        Tests the allocate slicing method.
        """
        self.controller.spectrum_obj.spectrum_props.is_free = True  # Ensure spectrum is free
        mock_get_spectrum.return_value = None  # Ensure that the method is mocked correctly

        # Call the allocate slicing method
        self.controller._allocate_slicing(num_segments=2, mod_format='QPSK', path_list=['A', 'B'], bandwidth='50G')

        # Verify that the relevant methods were called
        mock_get_spectrum.assert_called()
        mock_allocate.assert_called()
        mock_update_req_stats.assert_called_with(bandwidth='50G')

    @patch('src.sdn_controller.SDNController.allocate')
    @patch('src.sdn_controller.SDNController._update_req_stats')
    @patch('src.routing.Routing.get_route')
    @patch('src.spectrum_assignment.SpectrumAssignment.get_spectrum')
    def test_handle_event_arrival(self, mock_allocate, mock_stats, mock_route, mock_spectrum):  # pylint: disable=unused-argument
        """
        Tests the handle event with an arrival request.
        """
        mock_route.return_value = None
        self.controller.route_obj.route_props.paths_matrix = [['A', 'B', 'C']]
        self.controller.route_obj.route_props.mod_formats_matrix = [['QPSK']]
        self.controller.route_obj.route_props.weights_list = [10]

        self.controller.spectrum_obj.spectrum_props.is_free = True
        mock_spectrum.return_value = None

        self.controller.handle_event(req_dict={}, request_type="arrival")

        mock_allocate.assert_called_once()
        self.assertTrue(self.controller.sdn_props.was_routed)


if __name__ == '__main__':
    unittest.main()
