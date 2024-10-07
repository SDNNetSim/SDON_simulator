# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.spectrum_assignment import SpectrumAssignment


# TODO: Add fixed grid tests
class TestSpectrumAssignment(unittest.TestCase):
    """Unit tests for SpectrumAssignment class."""

    def setUp(self):
        """Set up the necessary properties and objects for the tests."""
        cores_matrix = {
            'c': np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 2, -2],
                [1, -1, 0, 0, 0, 0, 3, 3, 3, -3],
                [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
                [0, 4, 4, 4, -4, 0, 0, 1, 1, 0],
            ])
        }
        engine_props = {
            'cores_per_link': 2,
            'guard_slots': 1,
            'snr_type': 'None',
            'band_list': ['c'],
            'allocation_method': 'first_fit',
            'fixed_grid': False,
        }
        sdn_props = MagicMock()
        sdn_props.net_spec_dict = {
            ('source', 'dest'): {'cores_matrix': cores_matrix},
            ('dest', 'source'): {'cores_matrix': cores_matrix}
        }
        sdn_props.mod_formats_dict = {
            '16QAM': {'slots_needed': 2},
            'QPSK': {'slots_needed': 3},
            '64QAM': {'slots_needed': 4}
        }

        route_obj = MagicMock
        route_obj.route_props = {
            'paths_matrix': [], 'mod_formats_matrix': [], 'weights_list': [],
            'connection_index': [],
            'path_index': [], 'input_power': 0.001, 'freq_spacing': 12500000000.0,
            'mci_worst': 6.334975555658596e-27, 'max_link_length': None, 'span_len': 100.0,
            'max_span': None
        }

        self.spec_assign = SpectrumAssignment(engine_props=engine_props, sdn_props=sdn_props,
                                              route_props=route_obj.route_props)
        self.spec_assign.spectrum_props.slots_needed = 2
        self.spec_assign.spectrum_props.path_list = ['source', 'dest']
        self.spec_assign.spectrum_props.cores_matrix = cores_matrix

    def test_allocate_best_fit(self):
        """Test the allocation of the best fit."""
        channels_list = [
            {'link': ('source', 'dest'), 'core': 0, 'channel': [1, 2, 3, 4, 5], 'band': 'c'},
            {'link': ('source', 'dest'), 'core': 1, 'channel': [5, 6, 7, 8], 'band': 'l'}
        ]
        channels_list = sorted(channels_list, key=lambda d: len(d['channel']))
        self.spec_assign.spec_help_obj.check_other_links = MagicMock(return_value=True)
        self.spec_assign._allocate_best_fit(channels_list)

        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 7)
        self.assertEqual(self.spec_assign.spectrum_props.core_num, 1)
        self.assertEqual(self.spec_assign.spectrum_props.curr_band, 'l')

    def test_setup_first_last(self):
        """Test setting up first and last fit."""
        self.spec_assign.spectrum_props.forced_core = 2
        core_matrix, core_list, _ = self.spec_assign._setup_first_last()
        self.assertEqual(core_list, [2])
        self.assertTrue(np.array_equal(core_matrix[0], [[0, 4, 4, 4, -4, 0, 0, 1, 1, 0]]))

        self.spec_assign.spectrum_props.forced_core = None
        self.spec_assign.engine_props['allocation_method'] = 'priority_first'
        _, core_list, _ = self.spec_assign._setup_first_last()
        self.assertEqual(core_list, [0, 2, 4, 1, 3, 5, 6])

        self.spec_assign.engine_props['allocation_method'] = 'default'
        _, core_list, _ = self.spec_assign._setup_first_last()
        self.assertEqual(core_list, list(range(0, self.spec_assign.engine_props['cores_per_link'])))

    def test_first_fit(self):
        """Test first fit allocation."""
        self.spec_assign.engine_props['allocation_method'] = 'first_fit'
        self.spec_assign.handle_first_last('first_fit')
        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 3)

    def test_last_fit(self):
        """Test last fit allocation."""
        self.spec_assign.engine_props['allocation_method'] = 'last_fit'
        self.spec_assign.handle_first_last('last_fit')
        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 5)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 8)

    def test_get_spectrum(self):
        """Test getting spectrum allocation."""
        with patch.object(self.spec_assign, '_get_spectrum', wraps=self.spec_assign._get_spectrum) as mock_get_spectrum, \
                patch.object(self.spec_assign.snr_obj, 'handle_snr', return_value=(True, 0.5)) as _:
            # Ensure is_free starts as False
            self.spec_assign.spectrum_props.is_free = False
            mod_format_list = ['QPSK', '16QAM']

            # Simulate getting the spectrum successfully
            self.spec_assign.get_spectrum(mod_format_list, slice_bandwidth=None)

            # Verify the method was called and check the outcomes
            self.assertTrue(mock_get_spectrum.called, "Expected '_get_spectrum' to be called.")
            self.assertEqual(mock_get_spectrum.call_count, 1, "Expected '_get_spectrum' to be called once.")

            # Check that the spectrum was successfully allocated
            self.assertTrue(self.spec_assign.spectrum_props.is_free,
                            "Expected spectrum allocation to be successful (is_free=True).")
            self.assertEqual(self.spec_assign.spectrum_props.modulation, 'QPSK',
                             "Expected modulation to be set to 'QPSK'.")

    def test_first_fit_guard_band_zero(self):
        """Test allocation when the guard band is zero."""
        self.spec_assign.engine_props['guard_slots'] = 0
        self.spec_assign.spectrum_props.slots_needed = 2
        self.spec_assign.handle_first_last('first_fit')
        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 1)

    def test_first_fit_single_slot_no_guard(self):
        """Test allocation when only one slot is needed and guard slots are zero."""
        self.spec_assign.engine_props['guard_slots'] = 0
        self.spec_assign.spectrum_props.slots_needed = 1
        self.spec_assign.handle_first_last('first_fit')
        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 0)

    def test_first_fit_single_slot_with_guard(self):
        """Test allocation when only one slot is needed and guard slots are non-zero."""
        self.spec_assign.engine_props['guard_slots'] = 1
        self.spec_assign.spectrum_props.slots_needed = 1
        self.spec_assign.handle_first_last('first_fit')
        self.assertEqual(self.spec_assign.spectrum_props.start_slot, 0)
        self.assertEqual(self.spec_assign.spectrum_props.end_slot, 2)


if __name__ == '__main__':
    unittest.main()
