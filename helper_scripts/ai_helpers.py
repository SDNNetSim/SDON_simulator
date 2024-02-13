from ai_scripts.q_learning import QLearning


class AIMethods:
    """
    Contains methods related to artificial intelligence for the simulator.
    """

    def __init__(self, engine_props: dict):
        """
        Initializes the AIMethods class.
        """
        # TODO: Utilize ai props? Only if constructor is too large...
        self.ai_props = None
        self.engine_props = engine_props
        self.algorithm = engine_props['ai_algorithm']
        self.sdn_props = None
        self.route_props = None
        self.spectrum_props = None

        self.episode = 0
        self.seed = None
        # An object for the chosen AI algorithm
        self.ai_obj = None

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        self.ai_obj.save_model()

    def _q_update_env(self, was_routed: bool):
        """
        Updates the Q-learning environment.
        """
        self.ai_obj.curr_episode = self.episode
        self.ai_obj.update_env(was_routed=was_routed)
        if self.engine_props['is_training']:
            numerator = self.ai_obj.q_props['epsilon'] - self.engine_props['epsilon_end']
            denominator = float(self.engine_props['num_requests'])
            decay_amount = numerator / denominator
            self.ai_obj.decay_epsilon(amount=decay_amount)

    def _q_spectrum(self):
        raise NotImplementedError

    def _q_core(self):
        return self.ai_obj.get_core(spectrum_props=self.spectrum_props)

    def _q_routing(self, sdn_props: dict, route_props: dict):
        return self.ai_obj.get_route(sdn_props=sdn_props, route_props=route_props)

    def _init_q_learning(self):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.
        """
        self.ai_obj.curr_episode = self.episode
        self.ai_obj.setup_env()

        # Load a pre-trained table or start a new one
        if self.engine_props['is_training']:
            self.ai_obj.save_model()
        else:
            self.ai_obj.load_model()

        self.ai_obj.set_seed(self.seed)

    def save(self):
        """
        Responsible for saving relevant information.
        """
        if self.algorithm == 'q_learning':
            self._q_save()

    def assign_spectrum(self):
        if self.algorithm == 'q_learning':
            self._q_spectrum()

    def assign_core(self):
        if self.algorithm == 'q_learning':
            self._q_core()

    def route(self):
        """
        Responsible for routing.
        """
        if self.algorithm == 'q_learning':
            self._q_routing(sdn_props=self.sdn_props, route_props=self.route_props)

    def reset_epsilon(self):
        """
        Resets the epsilon parameter for an iteration or simulation.
        """
        self.ai_obj.epsilon = self.engine_props['epsilon_start']

    def update(self, was_routed: bool):
        """
        Responsible for updating environment information.
        """
        if self.algorithm == 'q_learning':
            self._q_update_env(was_routed=was_routed)

    def setup(self):
        """
        Responsible for setting up available AI algorithms and their methods.
        """
        if self.algorithm == 'q_learning':
            self.ai_obj = QLearning(engine_props=self.engine_props)
            self._init_q_learning()
        elif self.algorithm == 'None' or self.algorithm is None:
            pass
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
