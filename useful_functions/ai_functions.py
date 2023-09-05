# Local application imports
from ai.q_learning import *  # pylint: disable=unused-wildcard-import


class AIMethods:
    """
    Contains methods related to artificial intelligence for the simulator.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AIMethods class.
        """
        self.algorithm = kwargs['properties']['ai_algorithm']

        self.seed = None
        # An object for the chosen AI algorithm
        self.ai_obj = None

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        self.ai_obj.save_table()

    def _q_update_env(self, routed: bool, iteration: int):
        """
        Updates the Q-learning environment.

        :param routed: A flag to determine if a request was routed or not.
        :type routed: bool

        :param iteration: The current iteration of the simulation.
        :type iteration: int
        """
        self.ai_obj.update_environment(routed=routed)

        # Decay epsilon for half of the iterations evenly each time
        if 1 <= iteration <= iteration // 2 and self.ai_obj.sim_type == 'train':
            decay_amount = (self.ai_obj.epsilon / (iteration // 2) - 1)
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
        return path

    def _init_q_learning(self, params: dict):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.
        """
        if params['curr_episode'] == 0:
            self.ai_obj = QLearning(params=params)
        else:
            self.ai_obj.curr_episode = params['curr_episode']

        self.ai_obj.setup_environment()

        # Load a pre-trained table or start a new one
        # TODO: Erlang a hard-coded value :(
        if self.ai_obj.sim_type == 'train' and params['erlang'] == 10 or params['erlang'] == 50:
            self.ai_obj.save_table()
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
            self._q_update_env(routed=kwargs['routed'], iteration=kwargs['iteration'])

    def setup(self, **kwargs):
        """
        Responsible for setting up available AI algorithms and their methods.
        """
        if self.algorithm == 'q_learning':
            self._init_q_learning(params=kwargs['params'])
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
