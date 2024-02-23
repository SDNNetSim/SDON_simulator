# pylint: disable=protected-access
# pylint: disable=unused-argument

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import networkx as nx

from ai_scripts.q_learning import QLearning


class TestQLearning(unittest.TestCase):
    """
    Tests the methods within the QLearning class.
    """

    def setUp(self):
        self.engine_props = {
            'erlang': 10,
            'cores_per_link': 2,
            'max_iters': 10,
            'discount_factor': 0.8,
            'learn_rate': 0.3
        }
        self.q_props = {
            'save_params_dict': {
                'engine_params_list': ['erlang'],
                'other_params_list': ['some_param']
            },
            'some_param': 5,
        }

        # Additional setup from suggestions
        self.ql_agent = QLearning(self.engine_props)
        self.ql_agent.q_props['rewards_dict'] = {'routes_dict': {'rewards': {}}}
        self.ql_agent.q_props['errors_dict'] = {'routes_dict': {'errors': {}}}
        self.ql_agent.q_props['sum_rewards_dict'] = {}
        self.ql_agent.q_props['sum_errors_dict'] = {}
        self.ql_agent.q_props['epsilon'] = 0.05
        self.ql_agent.curr_episode = 0

        self.ql_agent.source = 0
        self.ql_agent.core_index = 0
        self.ql_agent.destination = 1
        self.ql_agent.path_index = 0
        self.ql_agent.q_props['routes_matrix'] = {0: {1: [{'q_value': 5.0}]}}

        # Setup for testing _init_q_tables
        self.route_types = [('path', 'O'), ('q_value', 'f8')]
        self.core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.sample_topology = nx.Graph()
        self.sample_topology.add_edge(0, 1, length=2)
        self.sample_topology.add_edge(1, 2, length=3)
        self.ql_agent.engine_props['topology'] = self.sample_topology

        self.ql_agent.num_nodes = 3
        self.ql_agent.k_paths = 1
        self.ql_agent.engine_props['cores_per_link'] = 2  # Assuming

        self.ql_agent.q_props['routes_matrix'] = np.empty(
            (self.ql_agent.num_nodes, self.ql_agent.num_nodes, self.ql_agent.k_paths),
            dtype=self.route_types)
        self.ql_agent.q_props['cores_matrix'] = np.empty(
            (self.ql_agent.num_nodes, self.ql_agent.num_nodes, self.ql_agent.k_paths,
             self.engine_props['cores_per_link']), dtype=self.core_types)

        self.ql_agent.q_props['cores_matrix'][0][1][0] = np.array([(np.array([]), 0, 3.0), (np.array([]), 0, 8.5)],
                                                                  dtype=self.core_types)
        self.ql_agent.paths_list = [['0', '1']]
        self.ql_agent.q_props['routes_matrix'][0] = np.array([
            [(np.array(['0', '1']), 5.0)],
            [(np.array(['0', '2', '1']), 2.8)],
            [(np.array(['0', '1', '2']), 3.0)]],
            dtype=self.route_types
        )

    def test_set_seed(self):
        """
        Tests the set seed method.
        """
        seed_value = 42

        np.random.seed(seed_value)
        random_number1 = np.random.rand()

        np.random.seed(seed_value)
        random_number2 = np.random.rand()

        self.assertEqual(random_number1, random_number2)

    def test_decay_epsilon(self):
        """
        Tests the decay epsilon method.
        """
        ql_agent = QLearning(self.engine_props)
        initial_epsilon = 0.2
        ql_agent.q_props['epsilon'] = initial_epsilon

        ql_agent.decay_epsilon(amount=0.1)
        self.assertEqual(ql_agent.q_props['epsilon'], initial_epsilon - 0.1)

        ql_agent.q_props['epsilon'] = 0.05
        with self.assertRaises(ValueError) as received_error:
            print(received_error)
            ql_agent.decay_epsilon(0.1)

    @patch('os.path.join')
    @patch('builtins.open')
    def test_save_params(self, mock_open, mock_os_path_join):
        """
        Tests the save params method.
        """
        ql_agent = QLearning(self.engine_props)
        ql_agent.q_props = self.q_props
        save_dir = 'test_output'

        ql_agent._save_params(save_dir)
        mock_os_path_join.assert_called_once_with(save_dir, 'e10_params_c2.json')

    def test_update_stats_new_episode(self):
        """
        Tests the update stats method.
        """
        reward = 10.0
        td_error = 0.5
        stats_flag = 'my_flag'

        self.ql_agent._update_stats(reward, td_error, stats_flag)

        expected_state = {
            'rewards_dict': {'my_flag': {'rewards': {'0': [10.0]}}},
            'errors_dict': {'my_flag': {'errors': {'0': [0.5]}}},
            'sum_rewards_dict': {'0': 10.0},
            'sum_errors_dict': {'0': 0.5}
        }

        for state_key, obj in expected_state.items():
            self.assertDictEqual(obj, self.ql_agent.q_props[state_key])

    def test_update_stats_existing_episode(self):
        """
        Tests the update stats method.
        """
        stats_flag = 'my_flag'
        initial_reward = 5.0
        initial_td_error = 0.2

        self.ql_agent.q_props = {
            'rewards_dict': {'my_flag': {'rewards': {'0': [initial_reward]}}},
            'errors_dict': {'my_flag': {'errors': {'0': [initial_td_error]}}},
            'sum_rewards_dict': {'0': initial_reward},
            'sum_errors_dict': {'0': initial_td_error}
        }

        new_reward = 12.0
        new_td_error = 0.8
        self.ql_agent._update_stats(new_reward, new_td_error, stats_flag)

        expected_state = {
            'rewards_dict': {'my_flag': {'rewards': {'0': [initial_reward, new_reward]}}},
            'errors_dict': {'my_flag': {'errors': {'0': [initial_td_error, new_td_error]}}},
            'sum_rewards_dict': {'0': initial_reward + new_reward},
            'sum_errors_dict': {'0': initial_td_error + new_td_error}
        }
        self.assertDictEqual(self.ql_agent.q_props, expected_state)

    @patch('ai_scripts.q_learning.QLearning._get_max_future_q')
    def test_update_routes_matrix_success(self, mock_get_max_future_q):
        """
        Tests the update routes method.
        """
        mock_get_max_future_q.return_value = 8.0
        self.ql_agent._update_routes_matrix(was_routed=True)
        expected_q_value = ((1.0 - 0.3) * 5.0) + (0.3 * (1.0 + 0.8 * 8.0))

        self.assertAlmostEqual(self.ql_agent.q_props['routes_matrix'][0][1][0]['q_value'], expected_q_value)

    @patch('ai_scripts.q_learning.QLearning._update_stats')
    def test_update_cores_matrix_success(self, mock_update_stats):
        """
        Tests the update cores method.
        """
        mock_update_stats.return_value = None

        initial_q = 2.5
        self.ql_agent.q_props['cores_matrix'] = {0: {1: [{0: {'q_value': initial_q}}]}}

        self.ql_agent._update_cores_matrix(was_routed=True)

        expected_q_value = (0.7 * initial_q) + (0.3 * (1.0 + 0.8 * 1.0))

        self.assertAlmostEqual(
            self.ql_agent.q_props['cores_matrix'][0][1][0][0]['q_value'],
            expected_q_value
        )

    @patch('networkx.shortest_simple_paths')
    def test_init_q_tables(self, mock_shortest_simple_paths):
        """
        Tests the init q-tables method.
        """
        mock_shortest_simple_paths.return_value = [[0, 1, 2]]

        self.ql_agent._init_q_tables()

        expected_route = (np.array([0, 1, 2]), 0.0)
        self.assertTrue(np.array_equal(self.ql_agent.q_props['routes_matrix'][0][2][0][0], expected_route[0]))

        expected_core_entry = (np.array([0, 1, 2]), 0, 0.0)
        self.assertTrue(np.array_equal(self.ql_agent.q_props['cores_matrix'][0][2][0][0][0], expected_core_entry[0]))

    def test_get_max_future_q(self):
        """
        Tests the get max future q method.
        """
        expected_max_q = 8.5
        result = self.ql_agent._get_max_future_q()
        self.assertEqual(result, expected_max_q)

    def test_get_max_curr_q(self):
        """
        Tests the max current q method.
        """
        expected_max_index = 0
        expected_max_path = ['0', '1']
        max_index, max_path = self.ql_agent._get_max_curr_q()
        self.assertEqual(max_index, expected_max_index)
        self.assertEqual(max_path, expected_max_path)

    @patch('numpy.random.uniform')
    @patch('numpy.random.choice')
    @patch('ai_scripts.q_learning.QLearning._update_route_props')
    def test_get_route(self, mock_random_choice, mock_random_uniform, mock_update_route_props):
        """
        Tests get route.
        """
        mock_random_uniform.return_value = 0.05
        mock_random_choice.return_value = 1
        mock_update_route_props.return_value = 100

        sdn_props = {'source': 0, 'destination': 1}
        route_props = MagicMock()
        self.ql_agent.paths_list = [['0', '2', '1']]

        self.ql_agent.get_route(sdn_props, route_props)
        self.assertTrue(np.array_equal(self.ql_agent.chosen_path, ['0', '2', '1']))

    @patch('numpy.random.uniform')
    @patch('numpy.random.randint')
    def test_get_core(self, mock_random_randint, mock_random_uniform):
        """
        Tests get core.
        """
        mock_random_uniform.return_value = 0.6
        self.ql_agent.q_props['cores_matrix'][0][1][0] = np.array([
            (np.array([]), 0, 7.0), (np.array([]), 0, 3.0)], dtype=self.core_types)

        spectrum_props = MagicMock()
        self.ql_agent.get_core(spectrum_props)

        self.assertEqual(self.ql_agent.core_index, 0)


if __name__ == '__main__':
    unittest.main()
