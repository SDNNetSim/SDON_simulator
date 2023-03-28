import json
import time
import threading

from handle_data.structure_data import structure_data
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir


# TODO: Support for run_arash method is currently outdated


class RunSim:
    """
    Controls all simulations for this project.
    """

    def __init__(self, mu=1.0, lam=2.0, num_requests=10000, max_iters=5, spectral_slots=256, num_cores=1,
                 # pylint: disable=invalid-name
                 bw_slot=12.5, max_lps=1, sim_flag='arash', constant_weight=True, guard_band=1):

        # Assumptions for things like mu, lambda, modulation format/calc, and routing
        self.sim_flag = sim_flag
        self.network_name = None
        # Determines whether the links all have a weight of one or their actual distances
        self.constant_weight = constant_weight
        self.seeds = list()
        self.num_requests = num_requests
        self.num_cores = num_cores
        self.mu = mu  # pylint: disable=invalid-name
        self.lam = lam
        # Amount of spectral slots allocated for the guard band
        self.guard_band = guard_band
        # Allocation policy
        self.allocation = 'first-fit'

        # Frequency for one spectrum slot (GHz)
        self.bw_slot = bw_slot
        # Maximum allowed light segment slicing (referred to as light path slicing)
        self.max_lps = max_lps
        self.bw_types = None
        self.req_dist = None
        self.spectral_slots = spectral_slots
        # Assign links with a numerical value for identification
        self.link_num = 1

        # If the confidence interval isn't reached, maximum allowed iterations
        self.max_iters = max_iters
        self.sim_input = None
        self.output_file_name = None
        # Thread number
        self.t_num = 1
        # Used to save simulation results
        self.sim_start = time.strftime("%m%d_%H:%M:%S")

    @staticmethod
    def save_input(file_name=None, obj=None):
        """
        Saves simulation input data. Not bandwidth data for now, since that is intended to be a constant and unchanged
        file (See create input).
        """
        create_dir('data/input')
        create_dir('data/output')

        with open(f'data/input/{file_name}', 'w', encoding='utf-8') as file_path:
            json.dump(obj, file_path, indent=4)

    def create_input(self):
        """
        Creates simulation input data.
        """
        bw_info = create_bw_info(assume=self.sim_flag)

        if self.t_num is None:
            file_name = 'bandwidth_info.json'
        else:
            file_name = f'bandwidth_info_{self.t_num}.json'

        self.save_input(file_name, bw_info)
        with open(f'./data/input/{file_name}', 'r',
                  encoding='utf-8') as fp:  # pylint: disable=invalid-name
            self.bw_types = json.load(fp)

        network_data = structure_data(constant_weight=self.constant_weight, network=self.network_name)
        physical_topology = create_pt(num_cores=self.num_cores, nodes_links=network_data)

        self.sim_input = {
            'seeds': self.seeds,
            'mu': self.mu,
            'lambda': self.lam,
            'number_of_request': self.num_requests,
            'bandwidth_types': self.bw_types,
            'max_lps': self.max_lps,
            'max_iters': self.max_iters,
            'spectral_slots': self.spectral_slots,
            'guard_band': self.guard_band,
            'physical_topology': physical_topology,
            'num_cores': self.num_cores,
            'allocation': self.allocation,
            'request_dist': self.req_dist,
        }

    def run_yue(self, max_lps=None, t_num=None, num_cores=1, allocation_method='first-fit', req_types=None):
        """
        Run the simulator based on Yue Wang's previous research assumptions. The paper can be found with this citation:
        Wang, Yue. Dynamic Traffic Scheduling Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss.
        University of Massachusetts Lowell, 2022.

        :param max_lps: The maximum allowed light path slicing for a given request
        :type max_lps: int
        :param t_num: The thread number or ID used to access files without locking
        :type t_num: int
        :param num_cores: The number of desired cores
        :type num_cores: int
        :param allocation_method: The spectral allocation policy
        :type allocation_method: str
        :param req_types: The distribution of the type of requests generated
        :type req_types: dict

        :return: None
        """
        self.mu = 0.2
        self.spectral_slots = 128
        self.sim_flag = 'yue'
        self.network_name = 'USNet'
        self.num_cores = num_cores
        self.constant_weight = False
        self.guard_band = 1
        self.allocation = allocation_method
        self.req_dist = req_types

        if max_lps is not None:
            self.max_lps = max_lps
            self.t_num = t_num
        else:
            raise NotImplementedError

        for lam in range(2, 143, 2):
            curr_erlang = float(lam) / self.mu
            lam *= float(self.num_cores)
            self.lam = float(lam)
            self.create_input()

            if self.t_num is None:
                file_name = 'simulation_input.json'
            else:
                file_name = f'simulation_input_{self.t_num}.json'

            self.save_input(file_name=file_name, obj=self.sim_input)
            engine = Engine(self.sim_input, erlang=curr_erlang, network_name=self.network_name,
                            sim_start=self.sim_start, assume=self.sim_flag,
                            sim_input_fp=f'./data/input/{file_name}', t_num=self.t_num)
            engine.run()

    def run_arash(self):
        """
        Run the simulator based on Arash Rezaee's previous research assumptions. The paper can be found with this
        citation: https://doi.org/10.1016/j.comnet.2020.107755

        :return: None
        """
        self.mu = 3600.0
        self.spectral_slots = 256
        self.sim_flag = 'arash'
        self.network_name = 'Pan-European'
        self.constant_weight = True
        self.guard_band = 0
        erlang_lst = [float(erlang) for erlang in range(50, 850, 50)]

        for curr_erlang in erlang_lst:
            self.lam = self.mu * float(self.num_cores) * curr_erlang
            self.create_input()

            self.save_input(file_name='simulation_input.json', obj=self.sim_input)

            engine = Engine(self.sim_input, erlang=curr_erlang, network_name=self.network_name,
                            sim_start=self.sim_start,
                            assume=self.sim_flag)
            engine.run()


def thread_sims():
    """
    Responsible for running the simulations in parallel.

    :return: None
    """
    tmp_list = list()
    for thread_num, thread_obj in tmp_list:
        curr_obj = RunSim()
        curr_thread = threading.Thread(target=curr_obj.run_yue, args=(
            thread_num, thread_obj['max_lps'], thread_obj['num_cores'], thread_obj['allocation'],
            thread_obj['req_types']))

        curr_thread.start()
        # Due to the simulations being saved under a directory by its start time
        if thread_obj['sleep']:
            time.sleep(2)


if __name__ == '__main__':
    thread_sims()
