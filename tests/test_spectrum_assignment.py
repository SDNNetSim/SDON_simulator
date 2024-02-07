# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock

import numpy as np

from sim_scripts.spectrum_assignment import SpectrumAssignment


class TestSpectrumAssignment(unittest.TestCase):
    """
    Test spectrum_assignment.py
    """

    def setUp(self):
        cores_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 2, -2],
            [1, -1, 0, 0, 0, 0, 3, 3, 3, -3],
            [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
            [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
            [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
            [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
            [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
        ])
        engine_props = {'cores_per_link': 2, 'guard_slots': 1, 'snr_type': 'None'}
        sdn_props = {
            'net_spec_dict': {
                ('source', 'dest'): {'cores_matrix': np.zeros((3, 10))}
            },
            'mod_formats': {'16QAM': {'slots_needed': 2}, 'QPSK': {'slots_needed': 3}, '64QAM': {'slots_needed': 4}},
        }
        self.spec_assign = SpectrumAssignment(engine_props=engine_props, sdn_props=sdn_props)
        self.spec_assign.spectrum_props.update({'slots_needed': 2, 'path_list': ['source', 'dest']})
        self.spec_assign.sdn_props['net_spec_dict'][('source', 'dest')]['cores_matrix'] = cores_matrix
        self.spec_assign.spectrum_props['cores_matrix'] = cores_matrix

    def test_allocate_best_fit(self):
        """
        Test allocate the best fit.
        """
        channels_list = [
            {'link': ('source', 'dest'), 'core': 0, 'channel': [1, 2, 3, 4, 5]},
            {'link': ('source', 'dest'), 'core': 1, 'channel': [5, 6, 7, 8]}
        ]
        self.spec_assign.spec_help_obj.check_other_links = MagicMock(return_value=True)
        self.spec_assign._allocate_best_fit(channels_list)

        self.assertEqual(self.spec_assign.spectrum_props['start_slot'], 1)
        self.assertEqual(self.spec_assign.spectrum_props['end_slot'], 4)
        self.assertEqual(self.spec_assign.spectrum_props['core_num'], 0)

    def test_find_best_fit(self):
        """
        Test find the best fit.
        """
        self.spec_assign.find_best_fit()

        self.assertEqual(self.spec_assign.spectrum_props['start_slot'], 2)
        self.assertEqual(self.spec_assign.spectrum_props['end_slot'], 5)
        self.assertEqual(self.spec_assign.spectrum_props['core_num'], 1)

    def test_setup_first_last(self):
        """
        Test setting up first and last fit.
        """
        self.spec_assign.spectrum_props['forced_core'] = 2
        core_matrix, core_list = self.spec_assign._setup_first_last()
        self.assertEqual(core_list, [2])
        self.assertTrue(np.array_equal(core_matrix[0], self.spec_assign.spectrum_props['cores_matrix'][2]))

        self.spec_assign.spectrum_props['forced_core'] = None
        self.spec_assign.engine_props['allocation_method'] = 'priority_first'
        _, core_list = self.spec_assign._setup_first_last()
        self.assertTrue(np.array_equal(core_list, [0, 2, 4, 1, 3, 5, 6]))

        self.spec_assign.engine_props['allocation_method'] = 'default'
        _, core_list = self.spec_assign._setup_first_last()
        self.assertEqual(core_list, list(range(0, self.spec_assign.engine_props['cores_per_link'])))

    def test_first_fit(self):
        """
        Test first fit.
        """
        self.spec_assign.engine_props['allocation_method'] = 'first_fit'
        self.spec_assign.handle_first_last('first_fit')
        self.assertEqual(self.spec_assign.spectrum_props['start_slot'], 0)
        self.assertEqual(self.spec_assign.spectrum_props['end_slot'], 3)

    def test_last_fit(self):
        """
        Test last fit.
        """
        self.spec_assign.engine_props['allocation_method'] = 'last_fit'
        self.spec_assign.handle_first_last('last_fit')
        self.assertEqual(self.spec_assign.spectrum_props['start_slot'], 5)
        self.assertEqual(self.spec_assign.spectrum_props['end_slot'], 8)

    def test_get_spectrum(self):
        """
        Test get spectrum.
        """
        self.spec_assign._init_spectrum_info = MagicMock()
        self.spec_assign._get_spectrum = MagicMock()
        self.spec_assign.snr_obj.handle_snr = MagicMock(return_value=(True, 0.5))

        mod_format_list = ['QPSK', '16QAM']
        self.spec_assign.get_spectrum(mod_format_list)
        self.assertTrue(self.spec_assign.spectrum_props['is_free'])
        self.assertEqual(self.spec_assign.spectrum_props['modulation'], 'QPSK')

        self.spec_assign.spectrum_props['is_free'] = False
        self.spec_assign._get_spectrum.side_effect = lambda: self.spec_assign.spectrum_props.update({'is_free': False})
        self.spec_assign.get_spectrum(mod_format_list)
        self.assertFalse(self.spec_assign.spectrum_props['is_free'])
        self.assertEqual(self.spec_assign.spectrum_props['block_reason'], 'congestion')

        self.spec_assign.spectrum_props['is_free'] = False
        self.spec_assign._get_spectrum.reset_mock(side_effect=True)
        slice_bandwidth = '50GHz'
        self.spec_assign.engine_props['mod_per_bw'] = {
            '50GHz': {
                'QPSK': {'slots_needed': 50},
                '16QAM': {'slots_needed': 100}
            }
        }
        self.spec_assign._get_spectrum.side_effect = lambda: self.spec_assign.spectrum_props.update({'is_free': True})
        self.spec_assign.get_spectrum(mod_format_list, slice_bandwidth)
        self.assertTrue(self.spec_assign.spectrum_props['is_free'])
        self.assertEqual(self.spec_assign.spectrum_props['slots_needed'], 50)
        mod_format_list_with_false = [False, 'QPSK']
        self.spec_assign.get_spectrum(mod_format_list_with_false)
        self.assertEqual(self.spec_assign.sdn_props['block_reason'], 'distance')


if __name__ == '__main__':
    unittest.main()
