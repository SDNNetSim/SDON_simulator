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
        self.max_segments = kwargs['properties']['max_segments']
        self.cores_per_link = kwargs['properties']['cores_per_link']
        self.is_training = kwargs['properties']['is_training']
        self.topology_info = kwargs['properties']['topology_info']
        self.topology = kwargs['properties']['topology']
        self.sim_info = kwargs['sim_info']
        self.seed = None

        # An object for the chosen AI algorithm
        self.ai_obj = None

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        self.ai_obj.save_table(path=self.sim_info, cores_per_link=self.cores_per_link)

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
        if 1 <= iteration <= iteration // 2 and self.is_training:
            decay_amount = (self.ai_obj.epsilon / (iteration // 2) - 1)
            self.ai_obj.decay_epsilon(amount=decay_amount)

    def _q_spectrum(self):
        raise NotImplementedError

    def _q_routing(self, source: str, destination: str):
        """
        Given a request, determines the path from source to destination.

        :param source: The source node.
        :type source: str

        :param destination: The destination node.
        :type destination: str

        :return: A path from source to destination and a modulation format.
        :rtype: list, str
        """
        self.ai_obj.source = source
        self.ai_obj.destination = destination
        path = self.ai_obj.route()
        return path

    def _init_q_learning(self, erlang: int = None, epsilon: float = 0.2, episodes: int = 1000, learn_rate: float = 0.5,
                         discount: float = 0.2, trained_table: str = None):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.

        :param erlang: The current load of the network.
        :type erlang: int

        :param epsilon: Parameter in the bellman equation to determine degree of randomness.
        :type epsilon: float

        :param episodes: The number of iterations or simulations.
        :type episodes: int

        :param learn_rate: The learning rate, alpha, in the bellman equation.
        :type learn_rate: float

        :param discount: The discount factor in the bellman equation to determine the balance between current and
                         future rewards.
        :type discount: float

        :param trained_table: A path to an already trained Q-table.
        :type trained_table: str
        """
        self.ai_obj = QLearning(epsilon=epsilon, episodes=episodes, learn_rate=learn_rate, discount=discount,
                                topology=self.topology)
        self.ai_obj.setup_environment()

        # Load a pre-trained table or start a new one
        # TODO: Erlang a hard-coded value :(
        if self.is_training and erlang == 10 or erlang == 50:
            self.ai_obj.save_table(path=self.sim_info, cores_per_link=self.cores_per_link)
        else:
            self.ai_obj.load_table(path=trained_table, cores_per_link=self.cores_per_link)

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
            resp = self._q_routing(source=kwargs['source'], destination=kwargs['destination'])

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
            self._init_q_learning(erlang=kwargs['erlang'], trained_table=kwargs['trained_table'])
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
