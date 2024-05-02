class PathAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    # TODO: These will move to multi-agent helpers script
    def _ql_route(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        routes_matrix = self.q_props['routes_matrix']
        # TODO: May need to modify this
        self.rl_props['paths_list'] = routes_matrix[self.rl_props['source']][self.rl_props['destination']]['path']

        self.paths_obj = self.helper_obj.classify_paths(paths_list=self.rl_props['paths_list'])
        if self.rl_props['paths_list'].ndim != 1:
            self.rl_props['paths_list'] = self.rl_props['paths_list'][:, 0]

        if random_float < self.q_props['epsilon']:
            self.rl_props['path_index'] = np.random.choice(self.rl_props['k_paths'])
            # The level will always be the last index
            self.level_index = self.paths_obj[self.rl_props['path_index']][-1]

            if self.rl_props['path_index'] == 1 and self.rl_props['k_paths'] == 1:
                self.rl_props['path_index'] = 0
            self.rl_props['chosen_path'] = self.rl_props['paths_list'][self.rl_props['path_index']]
        else:
            self.rl_props['path_index'], self.rl_props['chosen_path'] = self.helper_obj.get_max_curr_q(
                paths_info=self.paths_obj)
            self.level_index = self.paths_obj[self.rl_props['path_index']][-1]

        if len(self.rl_props['chosen_path']) == 0:
            raise ValueError('The chosen path can not be None')

        try:
            req_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count']]
        except:
            req_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count'] - 1]
        self.helper_obj.update_route_props(bandwidth=req_dict['bandwidth'], chosen_path=self.rl_props['chosen_path'])

    # TODO: To call the above function
    def get_route(self):
        raise NotImplementedError


class CoreAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    # TODO: Here or in the ql_agent script?
    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        if random_float < self.q_props['epsilon']:
            self.rl_props['core_index'] = np.random.randint(0, self.engine_obj.engine_props['cores_per_link'])
        else:
            cores_matrix = self.q_props['cores_matrix'][self.rl_props['source']][self.rl_props['destination']]
            q_values = cores_matrix[self.rl_props['path_index']]['q_value']
            self.rl_props['core_index'] = np.argmax(q_values)

    # TODO: To call the above function
    def get_core(self):
        raise NotImplementedError


class SpectrumAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_spectrum(self):
        raise NotImplementedError
