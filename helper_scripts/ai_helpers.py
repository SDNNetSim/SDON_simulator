class AIMethods:
    """
    Contains methods related to artificial intelligence for the simulator.
    """

    def __init__(self, engine_props: dict):
        """
        Initializes the AIMethods class.
        """
        self.ai_props = None
        self.engine_props = engine_props
        self.algorithm = engine_props['ai_algorithm']

        self.episode = 0
        self.seed = None
        # An object for the chosen AI algorithm
        self.ai_obj = None

        self._setup()

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        raise NotImplementedError

    def _q_update_env(self):
        """
        Updates the Q-learning environment.
        """
        raise NotImplementedError

    def _q_spectrum(self):
        raise NotImplementedError

    def _q_routing(self):
        raise NotImplementedError

    def _init_q_learning(self):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.
        """
        raise NotImplementedError

    def save(self):
        """
        Responsible for saving relevant information.
        """
        if self.algorithm == 'q_learning':
            self._q_save()

    def route(self):
        """
        Responsible for routing.
        """
        resp = None
        if self.algorithm == 'q_learning':
            resp = self._q_routing()

        return resp

    def reset_epsilon(self):
        """
        Resets the epsilon parameter for an iteration or simulation.
        """
        self.ai_obj.epsilon = self.engine_props['ai_arguments']['epsilon']

    def update(self):
        """
        Responsible for updating environment information.
        """
        if self.algorithm == 'q_learning':
            self._q_update_env()

    def _setup(self):
        """
        Responsible for setting up available AI algorithms and their methods.
        """
        if self.algorithm == 'q_learning':
            self.ai_obj = None
            self._init_q_learning()
        elif self.algorithm == 'None':
            pass
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
