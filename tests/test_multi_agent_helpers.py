import unittest
from unittest.mock import MagicMock, patch
from helper_scripts.multi_agent_helpers import PathAgent


class TestPathAgent(unittest.TestCase):
    """
    Test the path agent in multi_agent_helpers.py
    """

    def setUp(self):
        """
        Set up common test variables and mock objects.
        """
        self.rl_props = MagicMock()
        self.rl_help_obj = MagicMock()
        self.engine_props = {
            'reward': 10,
            'penalty': -5,
            'path_algorithm': 'q_learning',
            'k_paths': 3,
            'path_levels': 2,
            'epsilon_start': 0.01,
            'cores_per_link': 4,
        }
        self.path_agent = PathAgent(path_algorithm='q_learning', rl_props=self.rl_props, rl_help_obj=self.rl_help_obj)
        self.path_agent.engine_props = self.engine_props

    def test_end_iter_q_learning(self):
        """
        Test end_iter method when path_algorithm is 'q_learning'.
        """
        self.path_agent.agent_obj = MagicMock()
        self.path_agent.end_iter()
        self.path_agent.agent_obj.decay_epsilon.assert_called_once()

    def test_get_reward_allocated(self):
        """
        Test get_reward method when the request is allocated.
        """
        reward = self.path_agent.get_reward(was_allocated=True, path_length=5)
        self.assertEqual(reward, 10)

    def test_get_reward_not_allocated(self):
        """
        Test get_reward method when the request is not allocated.
        """
        reward = self.path_agent.get_reward(was_allocated=False, path_length=5)
        self.assertEqual(reward, -10)

    def test_update_q_learning(self):
        """
        Test update method for 'q_learning' algorithm.
        """
        self.path_agent.path_algorithm = 'q_learning'
        self.path_agent.agent_obj = MagicMock()
        self.path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
        self.path_agent.agent_obj.update_routes_matrix.assert_called_once()

    def test_update_epsilon_greedy_bandit(self):
        """
        Test update method for 'epsilon_greedy_bandit' algorithm.
        """
        self.path_agent.path_algorithm = 'epsilon_greedy_bandit'
        self.path_agent.agent_obj = MagicMock()
        self.path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
        self.path_agent.agent_obj.update.assert_called_once()

    def test_get_route_q_learning(self):
        """
        Test get_route method for 'q_learning' algorithm.
        """
        self.path_agent.path_algorithm = 'q_learning'
        with patch.object(self.path_agent, '_ql_route') as mock_ql_route:
            self.path_agent.get_route()
            mock_ql_route.assert_called_once()

    def test_get_route_bandit(self):
        """
        Test get_route method for bandit algorithms.
        """
        self.path_agent.path_algorithm = 'epsilon_greedy_bandit'
        with patch.object(self.path_agent, '_bandit_route') as mock_bandit_route:
            self.path_agent.get_route(route_obj=MagicMock())
            mock_bandit_route.assert_called_once()

    def test_load_model_q_learning(self):
        """
        Test load_model method when path_algorithm is 'q_learning'.
        """
        self.engine_props['path_algorithm'] = 'q_learning'
        with patch('numpy.load') as mock_load:
            self.path_agent.load_model(model_path='model', erlang=10.0, num_cores=4)
            mock_load.assert_called_once()

    def test_raise_not_implemented(self):
        """
        Test to ensure NotImplementedError is raised for unknown path_algorithm.
        """
        self.path_agent.path_algorithm = 'unknown_algorithm'
        with self.assertRaises(NotImplementedError):
            self.path_agent.setup_env()


if __name__ == '__main__':
    unittest.main()
