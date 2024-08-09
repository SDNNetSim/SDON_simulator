import unittest
import os
from unittest.mock import patch, MagicMock
import numpy as np

from helper_scripts.multi_agent_helpers import PathAgent
from helper_scripts.ql_helpers import QLearningHelpers
from helper_scripts.bandit_helpers import EpsilonGreedyBandit, UCBBandit


class TestPathAgent(unittest.TestCase):
    def setUp(self):
        self.rl_props = MagicMock()
        self.rl_props.k_paths = 1
        self.rl_props.paths_list = [[0]]
        self.rl_props.chosen_path_index = None
        self.rl_props.chosen_path_list = None
        self.rl_props.num_nodes = 5  # Ensure this is an integer

        self.rl_help_obj = MagicMock()
        self.engine_props = {
            'reward': 10,
            'penalty': -5,
            'path_algorithm': 'q_learning',
            'path_levels': ['low', 'medium', 'high'],  # Added the missing key
            'epsilon_start': 0.9,  # Added the missing key
            'epsilon_decay': 0.99,  # Added the missing key
            'epsilon_min': 0.1  # Added the missing key
        }

    def test_load_model(self):
        with patch('numpy.load') as mock_load, patch(
                'helper_scripts.ql_helpers.QLearningHelpers') as MockQLearningHelpers:
            mock_agent = MockQLearningHelpers.return_value
            path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props

            model_path = 'test_model'
            erlang = 10.0
            num_cores = 2

            # Ensuring setup_env initializes agent_obj
            path_agent.setup_env = MagicMock()
            path_agent.setup_env.return_value = None
            path_agent.agent_obj = mock_agent

            path_agent.load_model(model_path, erlang, num_cores)

            path_agent.setup_env.assert_called_once()
            mock_load.assert_called_once_with(os.path.join('logs', model_path, 'e10.0_routes_c2.npy'),
                                              allow_pickle=True)
            self.assertEqual(path_agent.agent_obj, mock_agent)
            self.assertEqual(mock_agent.props.routes_matrix, mock_load.return_value)

    def test_end_iter_q_learning(self):
        path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
        path_agent.agent_obj = MagicMock()
        path_agent.end_iter()
        path_agent.agent_obj.decay_epsilon.assert_called_once()

    def test_setup_env_q_learning(self):
        with patch('helper_scripts.ql_helpers.QLearningHelpers') as MockQLearningHelpers:
            mock_agent = MockQLearningHelpers.return_value
            path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props

            # Ensure rl_props.num_nodes is explicitly set to an integer
            self.rl_props.num_nodes = 5
            mock_agent.props = MagicMock()
            mock_agent.props.num_nodes = 5

            path_agent.setup_env()
            MockQLearningHelpers.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props)
            mock_agent.setup_env.assert_called_once()
            self.assertEqual(path_agent.agent_obj, mock_agent)

    def test_setup_env_epsilon_greedy_bandit(self):
        with patch('helper_scripts.bandit_helpers.EpsilonGreedyBandit') as MockEpsilonGreedyBandit:
            mock_agent = MockEpsilonGreedyBandit.return_value
            path_agent = PathAgent('epsilon_greedy_bandit', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props
            path_agent.setup_env()
            MockEpsilonGreedyBandit.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props,
                                                            is_path=True)
            self.assertEqual(path_agent.agent_obj, mock_agent)

    def test_setup_env_ucb_bandit(self):
        with patch('helper_scripts.bandit_helpers.UCBBandit') as MockUCBBandit:
            mock_agent = MockUCBBandit.return_value
            path_agent = PathAgent('ucb_bandit', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props
            path_agent.setup_env()
            MockUCBBandit.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True)
            self.assertEqual(path_agent.agent_obj, mock_agent)

    def test_get_reward(self):
        path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
        path_agent.engine_props = self.engine_props

        reward = path_agent.get_reward(was_allocated=True, path_length=5)
        self.assertEqual(reward, 10)

        reward = path_agent.get_reward(was_allocated=False, path_length=5)
        self.assertEqual(reward, -10)

    def test_update_q_learning(self):
        with patch('helper_scripts.ql_helpers.QLearningHelpers') as MockQLearningHelpers:
            mock_agent = MockQLearningHelpers.return_value
            path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props
            path_agent.setup_env()

            path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
            mock_agent.update_routes_matrix.assert_called_once()

    def test_update_epsilon_greedy_bandit(self):
        with patch('helper_scripts.bandit_helpers.EpsilonGreedyBandit') as MockEpsilonGreedyBandit:
            mock_agent = MockEpsilonGreedyBandit.return_value
            path_agent = PathAgent('epsilon_greedy_bandit', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props
            path_agent.setup_env()

            path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
            mock_agent.update.assert_called_once()

    def test_update_ucb_bandit(self):
        with patch('helper_scripts.bandit_helpers.UCBBandit') as MockUCBBandit:
            mock_agent = MockUCBBandit.return_value
            path_agent = PathAgent('ucb_bandit', self.rl_props, self.rl_help_obj)
            path_agent.engine_props = self.engine_props
            path_agent.setup_env()

            path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
            mock_agent.update.assert_called_once()

    @patch('numpy.random.choice')
    def test_ql_route_random_choice(self, mock_random_choice):
        mock_random_choice.return_value = 0
        path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
        path_agent.agent_obj = MagicMock()
        path_agent.rl_props.k_paths = 1
        path_agent.rl_props.paths_list = [[0]]
        path_agent.cong_list = [[0]]
        path_agent._PathAgent__ql_route(0.1)

        self.assertEqual(path_agent.rl_props.chosen_path_index, 0)

    def test_get_route_q_learning(self):
        path_agent = PathAgent('q_learning', self.rl_props, self.rl_help_obj)
        path_agent.setup_env = MagicMock()
        path_agent._ql_route = MagicMock()
        path_agent.get_route()

        path_agent._ql_route.assert_called_once()

    def test_get_route_bandit(self):
        path_agent = PathAgent('epsilon_greedy_bandit', self.rl_props, self.rl_help_obj)
        path_agent.setup_env = MagicMock()
        path_agent._bandit_route = MagicMock()
        path_agent.get_route(route_obj=MagicMock())

        path_agent._bandit_route.assert_called_once()


if __name__ == '__main__':
    unittest.main()
