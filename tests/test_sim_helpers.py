import unittest
import copy

import networkx as nx
import numpy as np

from helper_scripts.sim_helpers import get_path_mod, find_max_path_len, sort_dict_keys, sort_nested_dict_vals
from helper_scripts.sim_helpers import find_path_len, find_path_cong, get_channel_overlaps, find_free_slots
from helper_scripts.sim_helpers import find_free_channels, find_taken_channels, snake_to_title, int_to_string
from helper_scripts.sim_helpers import dict_to_list, list_to_title


class TestGetPathMod(unittest.TestCase):
    """
    Test sim_helpers.py
    """

    def setUp(self):
        self.mods_dict = {
            'QPSK': {'max_length': 2000},
            '16-QAM': {'max_length': 1000},
            '64-QAM': {'max_length': 500}
        }

        self.topology = nx.Graph()
        self.topology.add_edge(1, 2, length=10)
        self.topology.add_edge(2, 3, length=20)
        self.topology.add_edge(1, 3, length=50)
        self.topology.add_edge(3, 4, length=15)

        self.net_spec_dict = {
            ('A', 'B'): {'cores_matrix': np.array([[0, 1, 0, 0, 1],
                                                   [0, 0, 1, 0, 1]])},
            ('D', 'E'): {'cores_matrix': [[0, 0, 1, 1, -1],
                                          [0, 0, 0, 0, 0]]}
        }

    def test_valid_path_length(self):
        """
        Tests finding a valid path length.
        """
        path_len = 800
        expected_mod = '16-QAM'
        chosen_mod = get_path_mod(self.mods_dict, path_len)
        self.assertEqual(chosen_mod, expected_mod, f"{expected_mod} should be chosen for path length {path_len}")

    def test_exceeding_path_length(self):
        """
        Test a path length that's too long.
        """
        path_len = 2500
        chosen_mod = get_path_mod(self.mods_dict, path_len)
        self.assertFalse(chosen_mod, f"No modulation format should be chosen for path length {path_len}")

    def test_longest_hop_path(self):
        """
        Tests find the longest path.
        """
        source, destination = 1, 3
        expected_length = 30
        max_path_len = find_max_path_len(source, destination, self.topology)
        self.assertEqual(max_path_len, expected_length,
                         f"The longest path length from {source} to {destination} should be {expected_length}")

    def test_sort_nested_dict_vals(self):
        """
        Test sort nested dict vals.
        """
        original_dict = {'item1': {'nested_key': 10}, 'item2': {'nested_key': 5}, 'item3': {'nested_key': 15}}
        expected_sorted_dict = {'item2': {'nested_key': 5}, 'item1': {'nested_key': 10}, 'item3': {'nested_key': 15}}
        sorted_dict = sort_nested_dict_vals(original_dict, 'nested_key')
        self.assertEqual(sorted_dict, expected_sorted_dict,
                         "The dictionary was not sorted correctly based on the nested key's value.")

    def test_sort_dict_keys(self):
        """
        Test sort dict keys.
        """
        dictionary = {'3': 'c', '1': 'a', '2': 'b'}
        expected_sorted_dict = {'3': 'c', '2': 'b', '1': 'a'}
        sorted_dict = sort_dict_keys(dictionary)
        self.assertEqual(sorted_dict, expected_sorted_dict, "The dictionary keys were not sorted in descending order.")

    def test_find_path_len(self):
        """
        Test find path length.
        """
        path_list = [1, 2, 3, 4]
        expected_path_len = 45

        calculated_path_len = find_path_len(path_list, self.topology)
        self.assertEqual(calculated_path_len, expected_path_len,
                         "Calculated path length does not match the expected value.")

    def test_find_path_cong(self):
        """
        Test find path congestion.
        """
        net_spec_dict = {
            (1, 2): {'cores_matrix': np.array([[0, 1, 1], [1, 1, 0]])},
            (2, 3): {'cores_matrix': np.array([[1, 0, 0], [0, 0, 1]])}
        }
        path_list = [1, 2, 3]
        expected_average_cong = (4 / 6 + 2 / 6) / 2

        calculated_average_cong = find_path_cong(path_list, net_spec_dict)
        self.assertAlmostEqual(calculated_average_cong, expected_average_cong, places=5,
                               msg="Calculated average congestion does not match the expected value.")

    def test_get_channel_overlaps_seven_cores(self):
        """
        Tests get channel overlaps.
        """
        mock_free_channels = {0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3], 4: [], 5: [1, 2, 3],
                              6: [1, 2, 3]}
        mock_free_slots = {'link1': {0: [1, 2, 3], 1: [], 2: [1, 2, 3], 3: [1, 2, 3], 4: [1, 2, 3], 5: [],
                                     6: []}}

        expected_output = {
            'non_over_dict': {0: [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], 1: [], 2: [], 3: [], 4: [],
                              5: [], 6: []},
            'overlapped_dict': {0: [], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3], 4: [], 5: [1, 2, 3], 6: [1, 2, 3]}
        }

        resp = get_channel_overlaps(mock_free_channels, mock_free_slots)
        self.assertEqual(resp, expected_output)

    def test_find_free_slots(self):
        """
        Test find free slots.
        """
        result1 = find_free_slots(self.net_spec_dict, ('A', 'B'))
        expected_result1 = {0: np.array([0, 2, 3]), 1: np.array([0, 1, 3])}
        for core, slots_list in expected_result1.items():
            self.assertTrue(np.array_equal(result1[core], slots_list))

    def test_find_free_channels(self):
        """
        Test find free channels.
        """
        slots_needed = 2
        result1 = find_free_channels(self.net_spec_dict, slots_needed, ('A', 'B'))
        expected_result1 = {0: [[2, 3]], 1: [[0, 1]]}
        self.assertEqual(result1, expected_result1)

    def test_find_taken_channels(self):
        """
        Test find taken channels.
        """
        result1 = find_taken_channels(copy.deepcopy(self.net_spec_dict), ('D', 'E'))
        expected_result1 = {0: [[1, 1]], 1: []}
        self.assertEqual(result1, expected_result1)

    def test_snake_to_title(self):
        """
        Test snake to title.
        """
        snake_str = "hello_world"
        result = snake_to_title(snake_str)
        self.assertEqual(result, "Hello World")

    def test_int_to_string(self):
        """
        Test int to string.
        """
        number = 1234567
        result = int_to_string(number)
        self.assertEqual(result, "1,234,567")

    def test_dict_to_list(self):
        """
        Test dict to list.
        """
        data_dict = {
            'item1': {'value': 10},
            'item2': {'value': 20},
            'item3': {'value': 30}
        }
        nested_key = 'value'
        result = dict_to_list(data_dict, nested_key)
        self.assertTrue(np.array_equal(result, [10, 20, 30]))

    def test_list_to_title(self):
        """
        Test list to title.
        """
        input_list = [["Alice"], ["Bob"], ["Charlie"]]
        result = list_to_title(input_list)
        self.assertEqual(result, "Alice, Bob & Charlie")
