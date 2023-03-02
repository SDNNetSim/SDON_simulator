import json
import time
import threading

from handle_data.structure_data import structure_data
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir


# TODO: Allow command line input
# TODO: Need a best-fit flag


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
        self.constant_weight = constant_weight
        self.seeds = list()
        self.num_requests = num_requests
        self.num_cores = num_cores
        self.mu = mu  # pylint: disable=invalid-name
        self.lam = lam
        self.guard_band = guard_band
        self.allocation = 'first_fit'

        self.sim_start = time.strftime("%m%d_%H:%M:%S")

        # Frequency for one spectrum slot (GHz)
        self.bw_slot = bw_slot
        # Maximum allowed light segment slicing (light path slicing)
        self.max_lps = max_lps
        self.bw_types = None
        self.spectral_slots = spectral_slots
        # Initialize first link number
        self.link_num = 1

        # If the confidence interval isn't reached, maximum allowed iterations
        self.max_iters = max_iters
        self.sim_input = None
        self.output_file_name = None
        # Thread number
        self.t_num = 1

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
        }

    def run_yue(self, max_lps=None, t_num=None):
        """
        Run the simulator based on Yue Wang's previous research assumptions. The paper can be found with this citation:
        Wang, Yue. Dynamic Traffic Scheduling Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss.
        University of Massachusetts Lowell, 2022.

        :param max_lps: The maximum allowed light path slicing for a given request
        :type max_lps: int
        :param t_num: The thread number or ID used to access files without locking
        :type t_num: int

        :return: None
        """
        self.mu = 0.2
        self.spectral_slots = 128
        self.sim_flag = 'yue'
        self.network_name = 'USNet'
        self.constant_weight = False
        self.guard_band = 1
        self.allocation = 'best-fit'

        if max_lps is not None:
            self.max_lps = max_lps
            self.t_num = t_num
        else:
            raise NotImplementedError

        # TODO: Thread your lambda?
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


if __name__ == '__main__':
    obj_one = RunSim()
    obj_two = RunSim()
    obj_three = RunSim()
    obj_four = RunSim()
    obj_five = RunSim()

    # t1 = threading.Thread(target=obj_one.run_yue, args=(1, None,))
    t1 = threading.Thread(target=obj_one.run_yue, args=(1, 1,))
    t1.start()

    t2 = threading.Thread(target=obj_two.run_yue, args=(2, 2,))
    t2.start()

    t3 = threading.Thread(target=obj_three.run_yue, args=(4, 3,))
    t3.start()

    t4 = threading.Thread(target=obj_four.run_yue, args=(8, 4,))
    t4.start()

    t5 = threading.Thread(target=obj_five.run_yue, args=(16, 5,))
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
