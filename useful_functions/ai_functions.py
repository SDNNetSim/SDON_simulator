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

    def _q_update_env(self, routed: bool, spectrum: dict, path_mod: str):
        """
        Updates the Q-learning environment.

        :param routed: A flag to determine if a request was routed or not.
        :type routed: bool

        :param spectrum: Relevant information related to the spectrum assignment.
        :type spectrum: dict
        """
        self.ai_obj.curr_episode = self.episode
        self.ai_obj.update_env(routed=routed, spectrum=spectrum)
        # Decay epsilon
        if self.ai_obj.sim_type == 'train':
            numerator = self.ai_obj.ai_arguments['epsilon'] - self.ai_obj.ai_arguments['epsilon_target']
            denominator = float(self.properties['max_iters'])
            decay_amount = numerator / denominator
            self.ai_obj.decay_epsilon(amount=decay_amount)

    def _q_spectrum(self):
        raise NotImplementedError

    def _q_routing(self, source: str, destination: str, net_spec_db: dict, chosen_bw: str):
        """
        Given a request, determines the path from source to destination.

        :param source: The source node.
        :type source: str

        :param destination: The destination node.
        :type destination: str

        :param net_spec_db: The network spectrum database.
        :type net_spec_db: dict

        :param chosen_bw: The bandwidth of the current request to be routed.
        :type chosen_bw: str

        :return: A path from source to destination and a modulation format.
        :rtype: list, str
        """
        self.ai_obj.net_spec_db = net_spec_db
        self.ai_obj.source = source
        self.ai_obj.destination = destination
        self.ai_obj.chosen_bw = chosen_bw
        path = self.ai_obj.route()
        core = self.ai_obj.core_assignment()
        return path, core

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
            self.ai_obj.load_table()

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
            resp = self._q_routing(source=kwargs['source'], destination=kwargs['destination'],
                                   net_spec_db=kwargs['net_spec_db'], chosen_bw=kwargs['chosen_bw'])

        return resp

    def update(self, **kwargs):
        """
        Responsible for updating environment information.
        """
        if self.algorithm == 'q_learning':
            self._q_update_env(routed=kwargs['routed'], spectrum=kwargs['spectrum'], path_mod=kwargs['path_mod'])

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
