import networkx as nx
import numpy as np

from arg_scripts.rl_args import empty_q_props


# TODO: Generalize as many functions as you can, probably will move QLearning to another script
# TODO: Probably need to standardize function names for this to be generalized
class QLearningHelpers:
    def __init__(self):
        self.props = empty_q_props

    def _init_q_tables(self):
        for source in range(0, self.rl_props['num_nodes']):
            for destination in range(0, self.rl_props['num_nodes']):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = nx.shortest_simple_paths(G=self.engine_obj.engine_props['topology'],
                                                          source=str(source), target=str(destination), weight='length')

                # TODO: Generalize this and when creating the table
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.rl_props['k_paths']:
                        break

                    for level_index in range(3):
                        self.q_props['routes_matrix'][source, destination, k, level_index] = (curr_path, 0.0)

                    # for core_action in range(self.engine_obj.engine_props['cores_per_link']):
                    #     self.q_props['cores_matrix'][source, destination, k, core_action] = (curr_path, core_action,
                    #                                                                          0.0)

    def get_max_future_q(self, path_list):
        q_values = list()
        new_cong = find_path_cong(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict)
        # TODO: Fix this
        self.new_cong_index = self.helper_obj._classify_cong(curr_cong=new_cong)
        path_index, path, _ = self.paths_obj[self.rl_props['path_index']]
        self.paths_obj[self.rl_props['path_index']] = (path_index, path, self.new_cong_index)

        for path_index, _, cong_index in self.paths_obj:
            curr_q = self.q_props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']][path_index][
                cong_index]['q_value']
            q_values.append(curr_q)

        return np.max(q_values)

    def _update_routes_matrix(self, was_routed: bool):
        if was_routed:
            reward = 1.0
        else:
            reward = -1.0

        routes_matrix = self.q_props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']]
        current_q = routes_matrix[self.rl_props['path_index']][self.level_index]['q_value']
        max_future_q = self.get_max_future_q(path_list=routes_matrix[self.rl_props['path_index']][0][0])
        delta = reward + self.engine_obj.engine_props['discount_factor'] * max_future_q
        td_error = current_q - (reward + self.engine_obj.engine_props['discount_factor'] * max_future_q)
        self.helper_obj.update_q_stats(reward=reward, stats_flag='routes_dict', td_error=td_error,
                                       iteration=self.iteration)

        engine_props = self.engine_obj.engine_props
        new_q = ((1.0 - engine_props['learn_rate']) * current_q) + (engine_props['learn_rate'] * delta)

        routes_matrix = self.q_props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']]
        routes_matrix[self.rl_props['path_index']]['q_value'] = new_q

    def get_max_curr_q(self, paths_info):
        q_values = list()
        for path_index, _, level_index in paths_info:
            routes_matrix = self.q_props['routes_matrix'][self.ai_props['source']][self.ai_props['destination']]
            curr_q = routes_matrix[path_index][level_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        max_path = self.ai_props['paths_list'][max_index]
        return max_index, max_path

    def _calc_q_averages(self, stats_flag: str, episode: str, iteration: int):
        len_rewards = len(self.q_props['rewards_dict'][stats_flag]['rewards'][episode])

        max_iters = self.engine_obj.engine_props['max_iters']
        num_requests = self.engine_obj.engine_props['num_requests']

        if iteration == (max_iters - 1) and len_rewards == num_requests:
            self.completed_sim = True
            rewards_dict = self.q_props['rewards_dict'][stats_flag]['rewards']
            errors_dict = self.q_props['errors_dict'][stats_flag]['errors']
            self.q_props['rewards_dict'][stats_flag] = calc_matrix_stats(input_dict=rewards_dict)
            self.q_props['errors_dict'][stats_flag] = calc_matrix_stats(input_dict=errors_dict)

            self.save_model()

    def update_q_stats(self, reward: float, td_error: float, stats_flag: str, iteration: int):
        """
        Updates data structures related to tracking q-learning algorithm statistics.

        :param reward: The current reward.
        :param td_error: The current temporal difference error.
        :param stats_flag: Flag to determine calculations for the path or core q-table.
        :param iteration: The current episode/iteration.
        """
        if self.completed_sim:
            return

        episode = str(iteration)
        if episode not in self.q_props['rewards_dict'][stats_flag]['rewards'].keys():
            self.q_props['rewards_dict'][stats_flag]['rewards'][episode] = [reward]
            self.q_props['errors_dict'][stats_flag]['errors'][episode] = [td_error]
            self.q_props['sum_rewards_dict'][episode] = reward
            self.q_props['sum_errors_dict'][episode] = td_error
        else:
            self.q_props['rewards_dict'][stats_flag]['rewards'][episode].append(reward)
            self.q_props['errors_dict'][stats_flag]['errors'][episode].append(td_error)
            self.q_props['sum_rewards_dict'][episode] += reward
            self.q_props['sum_errors_dict'][episode] += td_error

        self._calc_q_averages(stats_flag=stats_flag, episode=episode, iteration=iteration)

    def _save_params(self, save_dir: str):
        params_dict = dict()
        for param_type, params_list in self.q_props['save_params_dict'].items():
            for curr_param in params_list:
                if param_type == 'engine_params_list':
                    params_dict[curr_param] = self.engine_obj.engine_props[curr_param]
                else:
                    params_dict[curr_param] = self.q_props[curr_param]

        erlang = self.engine_obj.engine_props['erlang']
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        param_fp = f"e{erlang}_params_c{cores_per_link}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    # TODO: Save every 'x' iters
    def save_model(self):
        """
        Saves the current q-learning model.
        """
        date_time = os.path.join(self.engine_obj.engine_props['network'], self.engine_obj.engine_props['sim_start'])
        save_dir = os.path.join('logs', 'ql', date_time)
        create_dir(file_path=save_dir)

        erlang = self.engine_obj.engine_props['erlang']
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        route_fp = f"e{erlang}_routes_c{cores_per_link}.npy"
        core_fp = f"e{erlang}_cores_c{cores_per_link}.npy"

        for curr_fp in (route_fp, core_fp):
            save_fp = os.path.join(os.getcwd(), save_dir, curr_fp)

            if curr_fp.split('_')[1] == 'routes':
                np.save(save_fp, self.q_props['routes_matrix'])
            else:
                np.save(save_fp, self.q_props['cores_matrix'])

        self._save_params(save_dir=save_dir)

    def decay_epsilon(self, amount: float, iteration: int):
        """
        Decays epsilon for the q-learning algorithm every step.

        :param amount: Amount to decay by.
        :param iteration: The current iteration/episode.
        """
        self.q_props['epsilon'] -= amount
        if iteration == 0:
            self.q_props['epsilon_list'].append(self.q_props['epsilon'])

        if self.q_props['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.q_props['epsilon']}")
