import unittest
from unittest.mock import patch, mock_open, MagicMock

import os
import numpy as np

from helper_scripts.bandit_helpers import load_model, _get_base_fp, _save_model, save_model


class TestBanditHelpers(unittest.TestCase):
    """
    Unit tests for the functions in the bandit_helpers script.
    """

    @patch('builtins.open', new_callable=mock_open, read_data='{"state1": [1, 2, 3]}')
    def test_load_model(self, mock_file):
        """
        Test loading a model from a JSON file.
        """
        expected_output = {'state1': [1, 2, 3]}
        result = load_model('model.json')
        self.assertEqual(result, expected_output)
        mock_file.assert_called_once_with(os.path.join('logs', 'model.json'), 'r', encoding='utf-8')

    def test_get_base_fp(self):
        """
        Test the generation of the base file path based on input parameters.
        """
        self.assertEqual(_get_base_fp(True, 10.0, 2), "e10.0_routes_c2.npy")
        self.assertEqual(_get_base_fp(False, 10.0, 2), "e10.0_cores_c2.npy")

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model(self, mock_file, mock_json_dump):
        """
        Test saving a model to a JSON file.
        """
        state_values_dict = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}
        expected_output = {"('a', 'b')": [1, 2, 3], "('c', 'd')": [4, 5, 6]}
        _save_model(state_values_dict, 10.0, 2, 'save_dir', True)
        mock_file.assert_called_once_with(os.path.join(os.getcwd(), 'save_dir', 'state_vals_e10.0_routes_c2.json'), 'w',
                                          encoding='utf-8')
        mock_json_dump.assert_called_once_with(expected_output, mock_file())

    @patch('helper_scripts.bandit_helpers.create_dir')
    @patch('numpy.save')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_function(self, mock_np_save, mock_create_dir, mock_save_model):
        """
        Test the overall save_model function that saves both the rewards array and the model.
        """
        max_iters = 100
        num_requests = 10
        iteration = max_iters - 1

        engine_props = {
            'max_iters': max_iters,
            'num_requests': num_requests,
            'network': 'network',
            'date': '2024-01-01',
            'sim_start': 'start',
            'erlang': 10.0,
            'cores_per_link': 2
        }
        props = MagicMock()

        rewards_matrix = [[1] * num_requests for _ in range(iteration + 1)]
        props.rewards_matrix = rewards_matrix

        self_obj = MagicMock()
        self_obj.engine_props = engine_props
        self_obj.props = props
        self_obj.is_path = True
        self_obj.values = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}

        save_model(iteration, 'alg', self_obj)

        save_dir = os.path.join('logs', 'alg', 'network', '2024-01-01', 'start')
        mock_create_dir.assert_called_once_with(file_path=save_dir)

        rewards_fp = os.path.join(save_dir, 'rewards_e10.0_routes_c2.npy')
        abs_rewards_fp = os.path.join(os.getcwd(), rewards_fp)
        args, _ = mock_np_save.call_args
        np.testing.assert_array_equal(args[1], np.mean(props.rewards_matrix, axis=0))
        self.assertEqual(args[0], abs_rewards_fp)

        actual_call = mock_save_model.call_args
        expected_state_values_dict = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}
        actual_state_values_dict = actual_call[1]['state_values_dict']

        for key in expected_state_values_dict:  # pylint: disable=consider-using-dict-items
            np.testing.assert_array_equal(expected_state_values_dict[key], actual_state_values_dict[key])

        expected_call_kwargs = {
            'state_values_dict': expected_state_values_dict,
            'erlang': 10.0,
            'cores_per_link': 2,
            'save_dir': save_dir,
            'is_path': True
        }

        self.assertEqual(expected_call_kwargs['erlang'], actual_call[1]['erlang'])
        self.assertEqual(expected_call_kwargs['cores_per_link'], actual_call[1]['cores_per_link'])
        self.assertEqual(expected_call_kwargs['save_dir'], actual_call[1]['save_dir'])
        self.assertEqual(expected_call_kwargs['is_path'], actual_call[1]['is_path'])


if __name__ == '__main__':
    unittest.main()
