# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock
from helper_scripts.spectrum_helpers import SpectrumHelpers


class TestSpectrumHelpers(unittest.TestCase):
    """
    Test spectrum_helpers.py
    """

    def setUp(self):
        self.engine_props = {'guard_slots': 1, 'allocation_method': 'first_fit'}
        self.sdn_props = {
            'net_spec_dict': {
                ('A', 'B'): {'cores_matrix': [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]},
                ('B', 'A'): {'cores_matrix': [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]},
                ('C', 'D'): {'cores_matrix': [[0, 1, 0, 1, 0], [1, 0, 1, 1, 0]]},
                ('D', 'C'): {'cores_matrix': [[0, 1, 0, 1, 0], [1, 0, 1, 1, 0]]},
                ('B', 'C'): {'cores_matrix': [[0, 1, 0, 1, 0], [1, 0, 1, 1, 0]]},
                ('C', 'B'): {'cores_matrix': [[0, 1, 0, 1, 0], [1, 0, 1, 1, 0]]},
            }
        }
        self.spectrum_props = {'path_list': ['A', 'B'], 'forced_core': None, 'slots_needed': 2}
        self.helpers = SpectrumHelpers(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                       spectrum_props=self.spectrum_props)

    def test_check_free_spectrum(self):
        """
        Test check free spectrum.
        """
        self.helpers.start_index = 0
        self.helpers.end_index = 1
        self.helpers.core_num = 0
        self.assertTrue(self.helpers._check_free_spectrum(('A', 'B'), ('B', 'A')))

        self.helpers.core_num = 1
        self.assertFalse(self.helpers._check_free_spectrum(('C', 'D'), ('D', 'C')))

    def test_check_other_links(self):
        """
        Test check other links.
        """
        self.helpers._check_free_spectrum = MagicMock(side_effect=[True, False])
        self.helpers.check_other_links()
        self.assertTrue(self.helpers.spectrum_props['is_free'])

        self.helpers._check_free_spectrum.reset_mock(side_effect=True)
        self.helpers._check_free_spectrum.side_effect = [True, True]
        self.helpers.check_other_links()
        self.assertTrue(self.helpers.spectrum_props['is_free'])

    def test_update_spec_props(self):
        """
        Test update spec props.
        """
        self.helpers.spectrum_props['forced_core'] = 2
        self.helpers.start_index = 1
        self.helpers.end_index = 3
        props = self.helpers._update_spec_props()
        self.assertEqual(props['core_num'], 2)
        self.assertEqual(props['start_slot'], 1)
        self.assertEqual(props['end_slot'], 4)

        self.helpers.spectrum_props['forced_core'] = None
        self.helpers.engine_props['allocation_method'] = 'last_fit'
        props = self.helpers._update_spec_props()
        self.assertEqual(props['start_slot'], 3)
        self.assertEqual(props['end_slot'], 2)

    def test_find_link_inters_static(self):
        """
        Test find link inters static helper method.
        """
        info_dict = {
            'free_slots_dict': {('A', 'B'): {0: [1, 2, 3], 1: [4, 5, 6]}},
            'free_channels_dict': {('A', 'B'): {0: ['ch1', 'ch2'], 1: ['ch3', 'ch4']}},
            'slots_inters_dict': {},
            'channel_inters_dict': {}
        }
        SpectrumHelpers._find_link_inters(info_dict, ('A', 'B'))

        self.assertSetEqual(info_dict['slots_inters_dict'][0], {1, 2, 3})
        self.assertSetEqual(info_dict['channel_inters_dict'][0], {'ch1', 'ch2'})

    def test_find_link_inters(self):
        """
        Test find link inters.
        """
        self.helpers.find_free_slots = MagicMock(return_value={0: [1, 2, 3], 1: [4, 5, 6]})
        self.helpers.find_free_channels = MagicMock(return_value={0: ['ch1', 'ch2'], 1: ['ch3', 'ch4']})
        self.helpers.spectrum_props['path_list'] = ['A', 'B', 'C']

        info_dict = self.helpers.find_link_inters()

        self.assertIn(('A', 'B'), info_dict['free_slots_dict'])
        self.assertIn(('B', 'C'), info_dict['free_slots_dict'])
        self.assertIn(('A', 'B'), info_dict['free_channels_dict'])
        self.assertIn(('B', 'C'), info_dict['free_channels_dict'])

    def test_find_best_core(self):
        """
        Test find the best core.
        """
        self.helpers.find_link_inters = MagicMock(return_value={
            'channel_inters_dict': {0: ['ch1', 'ch2'], 1: ['ch3'], 2: ['ch4', 'ch5', 'ch6']},
            'free_slots_dict': {
                ('A', 'B'): {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9], 3: [7, 8, 9], 4: [7, 8, 9], 5: [7, 8, 9],
                             6: [7, 8, 9]}}
        })
        self.helpers.get_channel_overlaps = MagicMock(return_value={
            'non_over_dict': {0: ['ch1'], 1: ['ch3'], 2: []}
        })

        best_core = self.helpers.find_best_core()
        self.assertEqual(best_core, 1)
