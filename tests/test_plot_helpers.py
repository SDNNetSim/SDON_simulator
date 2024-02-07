import unittest
import json
import os
from unittest.mock import patch, mock_open

from helper_scripts.plot_helpers import PlotHelpers
from helper_scripts.plot_helpers import _not_filters, _or_filters, _and_filters, _check_filters, find_times


class TestPlotHelpers(unittest.TestCase):
    """
    Tests methods in the PlotHelpers class.
    """

    def setUp(self):
        self.plot_props = {
            'output_dir': '/path/to/output',
            'input_dir': '/path/to/input',
            'plot_dict': None
        }
        self.net_names_list = ['Network1', 'Network2']
        self.plot_helper = PlotHelpers(plot_props=self.plot_props, net_names_list=self.net_names_list)

    @staticmethod
    def listdir_side_effect(path):
        """
        Simulates reading simulation numbers and file names.
        """
        if 's1' in path:
            return ['10_erlang.json', '20_erlang.json']
        if 's2' in path:
            return ['100_erlang.json', '110_erlang.json']
        if 'Network1' in path or 'Network2' in path:
            return ['s1', 's2']

        return []

    @staticmethod
    def read_json_file_side_effect(file_path):  # pylint: disable=unused-argument
        """
        Opens a mock input file.
        """
        file_path = os.path.join('tests', 'fixtures', 'input_file.json')
        with open(file_path, encoding='utf-8') as file_obj:
            input_dict = json.load(file_obj)
        return input_dict

    @patch('helper_scripts.plot_helpers.PlotHelpers._read_json_file')
    @patch('helper_scripts.plot_helpers.os.path.join')
    @patch('helper_scripts.plot_helpers.os.listdir')
    def test_get_file_info(self, mock_listdir, mock_join, mock_read_json):
        """
        Tests the get file info method.
        """
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_listdir.side_effect = self.listdir_side_effect
        mock_read_json.side_effect = self.read_json_file_side_effect

        sims_info_dict = {
            'networks_matrix': [['Network1']],
            'dates_matrix': [['2023-01-01']],
            'times_matrix': [['12:00']],
            'sims_matrix': [['s1', 's2']]
        }

        self.plot_helper.get_file_info(sims_info_dict)

        self.assertIn('12:00', self.plot_helper.file_info)
        self.assertEqual(self.plot_helper.file_info['12:00']['network'], 'Network1')
        self.assertEqual(self.plot_helper.file_info['12:00']['date'], '2023-01-01')

        self.assertIn('s1', self.plot_helper.file_info['12:00']['sim_dict'])
        self.assertIn('s2', self.plot_helper.file_info['12:00']['sim_dict'])

        expected_files_s1 = ['10', '20']
        expected_files_s2 = ['100', '110']
        self.assertEqual(self.plot_helper.file_info['12:00']['sim_dict']['s1'], expected_files_s1)
        self.assertEqual(self.plot_helper.file_info['12:00']['sim_dict']['s2'], expected_files_s2)

        mock_join.assert_called()
        mock_listdir.assert_called()
        mock_read_json.assert_called()

    def test_not_filter_present(self):
        """
        Tests the not filter method with a not filter present.
        """
        filter_dict = {'not_filter_list': [['key1', 'value1']]}
        file_dict = {'key1': 'value1'}
        self.assertFalse(_not_filters(filter_dict, file_dict))

    def test_not_filter_absent(self):
        """
        Tests the not filter method with a not filter not present.
        """
        filter_dict = {'not_filter_list': [['key1', 'value1']]}
        file_dict = {'key1': 'value2'}
        self.assertTrue(_not_filters(filter_dict, file_dict))

    def test_or_filter_match(self):
        """
        Tests the or filter method with a match.
        """
        filter_dict = {'or_filter_list': [['key1', 'value1'], ['key2', 'value2']]}
        file_dict = {'key1': 'value3', 'key2': 'value2'}
        self.assertTrue(_or_filters(filter_dict, file_dict))

    def test_or_filter_nomatch(self):
        """
        Tests the or filter method with no match.
        """
        filter_dict = {'or_filter_list': [['key1', 'value1'], ['key2', 'value2']]}
        file_dict = {'key1': 'value3', 'key3': 'value2'}
        self.assertFalse(_or_filters(filter_dict, file_dict))

    def test_and_filter_allmatch(self):
        """
        Tests the and filter method with every filter matching.
        """
        filter_dict = {'and_filter_list': [['key1', 'value1'], ['key2', 'value2']]}
        file_dict = {'key1': 'value1', 'key2': 'value2'}
        self.assertTrue(_and_filters(filter_dict, file_dict))

    def test_and_filter_partialmatch(self):
        """
        Tests the and filter method with only one filter match.
        """
        filter_dict = {'and_filter_list': [['key1', 'value1'], ['key2', 'value2']]}
        file_dict = {'key1': 'value1', 'key2': 'value3'}
        self.assertFalse(_and_filters(filter_dict, file_dict))

    def test_check_filters_false(self):
        """
        Test check filters for a false keep config return.
        """
        filter_dict = {
            'and_filter_list': [['key1', 'value1']],
            'or_filter_list': [['key2', 'value3']]
        }
        file_dict = {'key1': 'value1', 'key2': 'value2'}
        self.assertFalse(_check_filters(file_dict, filter_dict))

    @patch('helper_scripts.plot_helpers.os.path.isdir')
    @patch('helper_scripts.plot_helpers.os.listdir')
    @patch('helper_scripts.plot_helpers.open', mock_open(read_data=json.dumps({'key': 'value'})), create=True)
    @patch('helper_scripts.plot_helpers._check_filters', return_value=True)
    def test_find_times(self, mock_check_filters, mock_listdir, mock_isdir):  # pylint: disable=unused-argument
        """
        Tests the find times method.
        """
        mock_isdir.return_value = True
        mock_listdir.side_effect = lambda path: ['time1'] if 'time' not in path else ['sim_input_s1.json']

        dates_dict = {'2022-01-01': 'network1'}
        filter_dict = {}

        expected = {
            'times_matrix': [['time1']],
            'sims_matrix': [['s1']],
            'networks_matrix': [['network1']],
            'dates_matrix': [['2022-01-01']]
        }

        result = find_times(dates_dict, filter_dict)
        self.assertEqual(expected, result)
