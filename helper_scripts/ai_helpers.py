# Local application imports
from ai.q_learning import *  # pylint: disable=unused-wildcard-import


class AIMethods:
    """
    Contains methods related to artificial intelligence for the simulator.
    """

    def __init__(self, properties: dict):
        """
        Initializes the AIMethods class.
        """
        self.properties = properties
        self.algorithm = properties['ai_algorithm']

        self.episode = 0
        self.seed = None
        # An object for the chosen AI algorithm
        self.ai_obj = None

        self._setup()

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        self.ai_obj.save_tables()

    def _q_update_env(self, routed: bool):
        """
        Updates the Q-learning environment.

        :param routed: A flag to determine if a request was routed or not.
        :type routed: bool
        """
        self.ai_obj.curr_episode = self.episode
        self.ai_obj.update_env(routed=routed)
        # Decay epsilon
        if self.ai_obj.sim_type == 'train':
            numerator = self.ai_obj.ai_arguments['epsilon'] - self.ai_obj.ai_arguments['epsilon_target']
            denominator = float(self.properties['num_requests'])
            decay_amount = numerator / denominator
            self.ai_obj.decay_epsilon(amount=decay_amount)

    def _q_spectrum(self):
        raise NotImplementedError

    # TODO: A core should not be selected in routing
    def _q_routing(self, sdn_props: dict, route_props: dict):
        self.ai_obj.net_spec_db = sdn_props['net_spec_db']
        self.ai_obj.source = route_props['source']
        self.ai_obj.destination = route_props['destination']
        self.ai_obj.chosen_bw = route_props['chosen_bw']
        path = self.ai_obj.route()
        core = self.ai_obj.core_assignment()

        # A path could not be found, assign None to path modulation
        if not selected_path:
            resp = [selected_path], [False], [False], [False]
        else:
            path_len = find_path_len(path=selected_path, topology=self.sdn_props['topology'])
            path_mod = [
                get_path_mod(mod_formats=engine_props['mod_per_bw'][sdn_props['chosen_bw']], path_len=path_len)]
            resp = [selected_path], [selected_core], [path_mod], [path_len]

        return [path], [path_mod]

    def _init_q_learning(self):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.
        """
        self.ai_obj.curr_episode = self.episode
        self.ai_obj.setup_env()

        # Load a pre-trained table or start a new one
        if self.ai_obj.sim_type == 'train':
            self.ai_obj.save_tables()
        else:
            self.ai_obj.load_tables()

        self.ai_obj.set_seed(self.seed)

    def save(self):
        """
        Responsible for saving relevant information.
        """
        if self.algorithm == 'q_learning':
            self._q_save()

    def route(self, **kwargs):
        """
        Responsible for routing.
        """
        resp = None
        if self.algorithm == 'q_learning':
            resp = self._q_routing(sdn_props=kwargs['sdn_props'], route_props=kwargs['route_props'])

        return resp

    def reset_epsilon(self):
        self.ai_obj.epsilon = self.properties['ai_arguments']['epsilon']

    def update(self, **kwargs):
        """
        Responsible for updating environment information.
        """
        if self.algorithm == 'q_learning':
            self._q_update_env(routed=kwargs['routed'])

    def _setup(self):
        """
        Responsible for setting up available AI algorithms and their methods.
        """
        # TODO: Didn't pass is_training here
        if self.algorithm == 'q_learning':
            self.ai_obj = QLearning(properties=self.properties)
            self._init_q_learning()
        elif self.algorithm == 'None':
            pass
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
