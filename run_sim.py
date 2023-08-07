# Standard library imports
import json
import time
from typing import List, Dict

# Third-party library imports
import concurrent.futures

# Local application imports
from handle_data.structure_data import create_network
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir


# TODO: Spectrum assignment and routing objects created each time is inefficient


class NetworkSimulator:
    """
    Controls all simulations for this project.
    """

    def __init__(self, sim_data: dict = None, seeds: List[int] = None, mod_per_bw: dict = None,
                 req_dist: List[dict] = None, sim_fp: str = None, sim_start: str = None, net_name: str = 'USNet',
                 hold_time_mean: float = 1.0, arr_rate_mean: float = 2.0, num_reqs: int = 1000, max_iters: int = 10,
                 spectral_slots: int = 256, cores_per_link: int = 1, bw_per_slot: float = 12.5, max_segments: int = 1,
                 sim_type: str = 'arash', const_weight: bool = True, guard_slots: int = 1,
                 alloc_method: str = 'first-fit', route_method: str = None, dynamic_lps: bool = False,
                 thread_num: int = 1):
        """
        Initializes the NetworkSimulator class.

        :param sim_data: The final structured simulation input data.
        :type sim_data: dict

        :param seeds: The seed or seeds to be used for random generation.
        :type seeds: List[int]

        :param mod_per_bw: The bandwidth types that make up the network, for example 50 or 200 Gbps, accompanied by
                           the information related to their modulation formats.
        :type mod_per_bw: dict

        :param req_dist: The distribution of the bandwidth types, for example 80% 50 Gbps and 20% 200 Gbps.
        :type req_dist: List[dict]

        :param sim_fp: The path to the input file for the simulation.
        :type sim_fp: str

        :param net_name: The name of the network to be used.
        :type net_name: str

        :param hold_time_mean: The mean holding time for each request in the network.
        :type hold_time_mean: float

        :param arr_rate_mean: The mean arrival rate for each request in the network.
        :type arr_rate_mean: float

        :param num_reqs: The total number of requests to be allocated in a simulation run.
        :type num_reqs: int

        :param max_iters: The maximum number of iterations allowed if the confidence interval isn't reached.
        :type max_iters: int

        :param spectral_slots: The amount of spectral slots per each core.
        :type spectral_slots: int

        :param cores_per_link: The number of cores for every link in the network.
        :type cores_per_link: int

        :param bw_per_slot: The frequency for one spectrum slot (GHz).
        :type bw_per_slot: float

        :param max_segments: The maximum allowed light segments a single request may be sliced into.
        :type max_segments: int

        :param sim_type: A flag to determine which simulation to run and with which assumptions for the simulation.
        :type sim_type: str

        :param const_weight: A flag that tells us if a link in the network has a value of the actual distance or a
                             constant value of one.
        :type const_weight: bool

        :param guard_slots: The amount of slots the guard band for a request will occupy.
        :type guard_slots: int

        :param alloc_method: The allocation policy for a request.
        :type alloc_method: str

        :param route_method: The routing policy for a request.
        :type route_method: str

        :param dynamic_lps: A flag to determine the type of light path slicing to be implemented. Here, we may slice a
                            request to multiple different bandwidths if set to true.
        :type dynamic_lps: bool

        :param thread_num: Used to identify simulations running in parallel.
        :type thread_num: int
        """
        self.sim_data = sim_data
        self.seeds = seeds
        self.net_name = net_name
        self.sim_type = sim_type
        self.const_weight = const_weight
        self.num_reqs = num_reqs
        self.cores_per_link = cores_per_link
        self.hold_time_mean = hold_time_mean
        self.arr_rate_mean = arr_rate_mean
        self.guard_slots = guard_slots
        self.alloc_method = alloc_method
        self.dynamic_lps = dynamic_lps

        self.sim_start = sim_start

        self.bw_per_slot = bw_per_slot
        self.max_segments = max_segments
        self.mod_per_bw = mod_per_bw
        self.req_dist = req_dist
        self.spectral_slots = spectral_slots

        self.max_iters = max_iters
        self.sim_fp = sim_fp
        self.thread_num = thread_num

        # The date and current time derived from the simulation start
        self.date, self.curr_time = self.sim_start.split('_')[0], self.sim_start.split('_')[1]

        # Used for machine and reinforcement learning
        self.ai_algorithm = None
        self.is_training = None
        self.train_file = None

        # Used for NLI routing calculations
        self.beta = None

    def save_input(self, file_name: str = None, data: Dict = None):
        """
        Saves simulation input data. Does not save bandwidth data, as that is intended to be a constant and unchanged
        file.

        :param file_name: The name of the file to save the input data to.
        :type file_name: str

        :param data: The data to save to the file.
        :type data: dict

        :return: None
        """
        create_dir(f'data/input/{self.net_name}/{self.date}/{self.curr_time}')
        create_dir('data/output')

        with open(f'data/input/{self.net_name}/{self.date}/{self.curr_time}/{file_name}', 'w',
                  encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def create_input(self):
        """
        Create the input data for the simulation.

        The method generates bandwidth information, creates the physical topology of the network,
        and creates a dictionary containing all the necessary simulation parameters.

        :return: None
        """
        bw_info = create_bw_info(sim_type=self.sim_type)

        if self.thread_num is None:
            bw_file = 'bw_info.json'
        else:
            bw_file = f'bw_info_{self.thread_num}.json'

        self.save_input(file_name=bw_file, data=bw_info)

        with open(f'./data/input/{self.net_name}/{self.date}/{self.curr_time}/{bw_file}', 'r',
                  encoding='utf-8') as file_object:
            self.mod_per_bw = json.load(file_object)

        network_data = create_network(const_weight=self.const_weight, net_name=self.net_name)
        topology = create_pt(cores_per_link=self.cores_per_link, network_data=network_data)

        self.sim_data = {
            'seeds': self.seeds,
            'hold_time_mean': self.hold_time_mean,
            'arr_rate_mean': self.arr_rate_mean,
            'num_reqs': self.num_reqs,
            'mod_per_bw': self.mod_per_bw,
            'max_segments': self.max_segments,
            'max_iters': self.max_iters,
            'spectral_slots': self.spectral_slots,
            'guard_slots': self.guard_slots,
            'topology': topology,
            'cores_per_link': self.cores_per_link,
            'alloc_method': self.alloc_method,
            'route_method': self.route_method,
            'req_dist': self.req_dist,
            'ai_algorithm': self.ai_algorithm,
            'is_training': self.is_training,
            'train_file': self.train_file,
            'beta': self.beta
        }

    def run_yue(self, max_segments, thread_num, cores_per_link, alloc_method, req_dist, dynamic_lps, ai_algorithm,
                is_training, train_file, max_iters, route_method, beta):
        """
        Runs a Yue-based simulation with the specified parameters. Reference: Wang, Yue. Dynamic Traffic Scheduling
        Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss. University of Massachusetts Lowell, 2022.

        :param max_segments: Maximum number of light segments one request may be broken into.
        :type max_segments: int

        :param thread_num: Unique identifier for the simulation.
        :type thread_num: int

        :param cores_per_link: Number of cores to use for the simulation.
        :type cores_per_link: int

        :param alloc_method: Method for allocating wavelengths.
        :type alloc_method: str

        :param req_dist: Distribution of requests to use.
        :type req_dist: str

        :param dynamic_lps: Flag to determine dynamic light path slicing ability for any given run.
        :type dynamic_lps: bool

        :param ai_algorithm: The type of artificial intelligence algorithm to be run.
        :type ai_algorithm: str

        :param: is_training: Determines if we are training or testing ML/RL methods.
        :type is_training: bool

        :param train_file: If not training, for testing we must have a trained algorithm to load.
        :type train_file: str

        :param max_iters: Determines the maximum number of iterations.
        :type max_iters: int

        :param route_method: The type of routing to be used for the SDN controller.
        :type route_method: str

        :param beta: Used for NLI routing calculation to consider importance of NLI impairment.
        :type beta: float

        :return: None
        """
        self.hold_time_mean = 0.2
        self.spectral_slots = 128
        self.sim_type = 'yue'
        self.net_name = 'USNet'
        self.const_weight = False
        self.guard_slots = 1
        self.cores_per_link = cores_per_link

        self.alloc_method = alloc_method
        self.route_method = route_method
        self.dynamic_lps = dynamic_lps
        self.req_dist = req_dist

        self.max_iters = max_iters
        self.max_segments = max_segments
        self.ai_algorithm = ai_algorithm
        self.is_training = is_training
        self.train_file = train_file
        self.thread_num = thread_num

        self.beta = beta

        for arr_rate_mean in range(2, 143, 2):
            erlang = float(arr_rate_mean) / self.hold_time_mean
            arr_rate_mean *= float(self.cores_per_link)
            self.arr_rate_mean = float(arr_rate_mean)
            self.create_input()

            if self.thread_num is None:
                file_name = 'sim_input.json'
            else:
                file_name = f'sim_input_{self.thread_num}.json'

            self.save_input(file_name=file_name, data=self.sim_data)
            engine = Engine(sim_data=self.sim_data, erlang=erlang, net_name=self.net_name,
                            sim_start=self.sim_start, sim_type=self.sim_type,
                            input_fp=f'./data/input/{self.net_name}/{self.date}/{self.curr_time}/{file_name}',
                            thread_num=thread_num, dynamic_lps=dynamic_lps, is_training=is_training)
            engine.run()

    # TODO: This method does not have support at this point in time
    def run_arash(self):
        """
        Runs a simulation using the Arash simulation flag. Reference: https://doi.org/10.1016/j.comnet.2020.107755.

        :return: None
        """
        self.arr_rate_mean = 3600.0
        self.spectral_slots = 256
        self.sim_type = 'arash'
        self.net_name = 'Pan-European'
        self.const_weight = True
        self.guard_slots = 0
        erlang_lst = [float(erlang) for erlang in range(50, 850, 50)]

        for erlang in erlang_lst:
            self.arr_rate_mean = self.hold_time_mean * float(self.cores_per_link) * erlang
            self.create_input()

            self.save_input(file_name='sim_input.json', data=self.sim_data)

            engine = Engine(sim_data=self.sim_data, erlang=erlang, net_name=self.net_name, sim_start=self.sim_start,
                            sim_type=self.sim_type)
            engine.run()


def run(threads):
    """
    Runs multiple simulations concurrently using threads.

    :param threads: A list of dictionaries, where each dictionary contains the parameters for a single simulation.
    :type threads: list of dicts

    :return: None
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for thread_num, thread_params in enumerate(threads, start=1):
            class_inst = NetworkSimulator(sim_start=time.strftime("%m%d_%H:%M:%S"))

            future = executor.submit(class_inst.run_yue, thread_params['max_segments'], thread_num,
                                     thread_params['cores_per_link'], thread_params['alloc_method'],
                                     thread_params['req_dist'], thread_params['dynamic_lps'],
                                     thread_params['ai_algorithm'], thread_params['is_training'],
                                     thread_params['train_file'], thread_params['max_iters'],
                                     thread_params['route_method'], thread_params['beta'])

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()


# TODO: Move to a configuration file (after yue and arash methods work)
if __name__ == '__main__':
    threads_obj = []
    for beta in [0.5]:
        for is_training in [True]:
            for max_iters in [5]:
                for dynamic_flag in [False]:
                    for max_segments in [1]:
                        for cores_per_link in [1]:
                            thread = {
                                'max_segments': max_segments,
                                'cores_per_link': cores_per_link,
                                'alloc_method': 'first-fit',
                                'req_dist': {'25': 0.0, '50': 0.0, '100': 0.5, '200': 0.0, '400': 0.5},
                                'dynamic_lps': dynamic_flag,
                                'ai_algorithm': None,
                                'route_method': 'nli_aware',
                                'is_training': is_training,
                                'train_file': None,
                                'max_iters': max_iters,
                                'beta': beta
                            }
                            threads_obj.append(thread)

    run(threads_obj)
