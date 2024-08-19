# pylint: disable=protected-access

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from helper_scripts.spectrum_helpers import SpectrumHelpers


class TestSpectrumHelpers(unittest.TestCase):
    """Unit tests for the SpectrumHelpers class."""

    def setUp(self):
        """Set up the test environment for SpectrumHelpers."""
        self.engine_props = {
            'allocation_method': 'first_fit',
            'guard_slots': 1
        }
        # Initialize spectrum_props as an object with attributes
        self.spectrum_props = SimpleNamespace(
            path_list=[1, 2, 3],
            slots_needed=2,
            is_free=False,
            forced_core=None,
            forced_band=None,
            forced_index=None,
            start_slot=None,
            end_slot=None,
            core_num=None,
            curr_band=None
        )
        self.sdn_props = MagicMock()
        # Initialize cores_matrix as a 2D array (matrix) with 2 cores and 10 slots each
        self.sdn_props.net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': np.zeros((2, 10))}},
            (2, 1): {'cores_matrix': {'c': np.zeros((2, 10))}},
            (2, 3): {'cores_matrix': {'c': np.zeros((2, 10))}},
            (3, 2): {'cores_matrix': {'c': np.zeros((2, 10))}},
        }
        self.helpers = SpectrumHelpers(self.engine_props, self.sdn_props, self.spectrum_props)

    def test_check_free_spectrum(self):
        """Test the _check_free_spectrum method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        result = self.helpers._check_free_spectrum((1, 2), (2, 3))
        self.assertTrue(result)

    def test_check_other_links(self):
        """Test the check_other_links method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        self.helpers.check_other_links()
        self.assertTrue(self.spectrum_props.is_free)

    def test_update_spec_props(self):
        """Test the _update_spec_props method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        self.helpers._update_spec_props()
        self.assertEqual(self.spectrum_props.start_slot, 0)
        self.assertEqual(self.spectrum_props.end_slot, 6)
        self.assertEqual(self.spectrum_props.core_num, 0)
        self.assertEqual(self.spectrum_props.curr_band, 'c')

    def test_check_super_channels(self):
        """Test checking for available super-channels."""
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]

        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0

        # Test when there is a valid allocation
        self.spectrum_props.slots_needed = 2
        self.helpers.engine_props['guard_slots'] = 1
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag=''))

        # Test when there is no valid allocation (e.g., due to forced index)
        self.spectrum_props.forced_index = 5
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag='forced_index'))

        # Test when the allocation method is last_fit
        self.helpers.engine_props['allocation_method'] = 'last_fit'
        self.assertFalse(self.helpers.check_super_channels(open_slots_matrix, flag=''))

        # Test when no super-channel can satisfy the request
        self.spectrum_props.slots_needed = 10
        self.assertFalse(self.helpers.check_super_channels(open_slots_matrix, flag=''))

    def test_find_best_core(self):
        """Test finding the best core with the least overlap in a seven-core setup."""
        # Set up the mock data for a seven-core system
        path_info = {
            'free_channels_dict': {
                (1, 2): {'c': {
                    0: [[1, 2, 3]], 1: [[4, 5, 6]], 2: [[7, 8, 9]],
                    3: [[10, 11, 12]], 4: [[13, 14, 15]], 5: [[16, 17, 18]],
                    6: [[19, 20, 21]]
                }},
                (2, 3): {'c': {
                    0: [[1, 2, 3]], 1: [[4, 5, 6]], 2: [[7, 8, 9]],
                    3: [[10, 11, 12]], 4: [[13, 14, 15]], 5: [[16, 17, 18]],
                    6: [[19, 20, 21]]
                }}
            },
            'free_slots_dict': {
                (1, 2): {'c': {
                    0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9],
                    3: [10, 11, 12], 4: [13, 14, 15], 5: [16, 17, 18],
                    6: [19, 20, 21]
                }},
                (2, 3): {'c': {
                    0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9],
                    3: [10, 11, 12], 4: [13, 14, 15], 5: [16, 17, 18],
                    6: [19, 20, 21]
                }}
            }
        }

        # Mock the `find_link_inters` method to return the above path_info
        self.helpers.find_link_inters = MagicMock(return_value=path_info)

        best_core = self.helpers.find_best_core()
        self.assertEqual(best_core, 0)


if __name__ == '__main__':
    unittest.main()
