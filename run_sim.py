import json
import time
import threading

from handle_data.structure_data import structure_data
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir


# TODO: Allow command line input


class RunSim:
    """
    Controls all simulations for this project.
    """

    def __init__(self, mu=1.0, lam=2.0, num_requests=10000, max_iters=1, spectral_slots=256, num_cores=1,
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
        self.req_dist = None
        self.cong_only = None
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
            'request_dist': self.req_dist,
            'cong_only': self.cong_only,
        }

    def run_yue(self, max_lps=None, t_num=None, num_cores=1, allocation_method='first-fit', req_dist=None,
                cong_only=None):
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
        :param req_dist: The distribution of the type of requests generated
        :type req_dist: dict
        :param cong_only: Whether to generate requests that are blocked ONLY due to congestion and not distance
        :type cong_only: bool

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
        self.req_dist = req_dist
        self.cong_only = cong_only

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


if __name__ == '__main__':
    """
    Simulations to be ran:
        - Number of requests vs. slots occupied info (1, 4, and 7 cores baseline) [12]
            - This will be more of a line graph
        - Number of times a request was sliced
            - This will be a histogram
            
            * For every request, have one large dictionary that documents:
                * The number of slots occupied in the network (could more cores be misleading here?)
                * The amount of times that request was sliced
            
        - Unlimited slicing
            - 1, 4, and 7 cores
                - All bandwidths (baseline) [12]
                - Individual bandwidths [36]
    """
    req_dist = {'25': 0.0, '50': 0.3, '100': 0.5, '200': 0.0, '400': 0.2}
    obj_one = RunSim()
    obj_two = RunSim()
    obj_three = RunSim()
    obj_four = RunSim()

    # Covers baseline single core
    t1 = threading.Thread(target=obj_one.run_yue, args=(1, 1, 1, 'first-fit', req_dist, False))
    t1.start()

    t2 = threading.Thread(target=obj_two.run_yue, args=(2, 2, 1, 'first-fit', req_dist, False))
    t2.start()

    t3 = threading.Thread(target=obj_three.run_yue, args=(4, 3, 1, 'first-fit', req_dist, False))
    t3.start()

    t4 = threading.Thread(target=obj_four.run_yue, args=(8, 4, 1, 'first-fit', req_dist, False))
    t4.start()

    time.sleep(2)

    obj_five = RunSim()
    obj_six = RunSim()
    obj_seven = RunSim()
    obj_eight = RunSim()

    # Covers baseline four cores
    t5 = threading.Thread(target=obj_five.run_yue, args=(1, 5, 4, 'first-fit', req_dist, False))
    t5.start()

    t6 = threading.Thread(target=obj_six.run_yue, args=(2, 6, 4, 'first-fit', req_dist, False))
    t6.start()

    t7 = threading.Thread(target=obj_seven.run_yue, args=(4, 7, 4, 'first-fit', req_dist, False))
    t7.start()

    t8 = threading.Thread(target=obj_eight.run_yue, args=(8, 8, 4, 'first-fit', req_dist, False))
    t8.start()

    time.sleep(2)

    obj_nine = RunSim()
    obj_ten = RunSim()
    obj_eleven = RunSim()
    obj_twelve = RunSim()

    # Covers baseline seven cores
    t9 = threading.Thread(target=obj_nine.run_yue, args=(1, 9, 7, 'first-fit', req_dist, False))
    t9.start()

    t10 = threading.Thread(target=obj_ten.run_yue, args=(2, 10, 7, 'first-fit', req_dist, False))
    t10.start()

    t11 = threading.Thread(target=obj_eleven.run_yue, args=(4, 11, 7, 'first-fit', req_dist, False))
    t11.start()

    t12 = threading.Thread(target=obj_twelve.run_yue, args=(8, 12, 7, 'first-fit', req_dist, False))
    t12.start()

    time.sleep(2)

    obj_thirteen = RunSim()
    obj_fourteen = RunSim()
    obj_fifteen = RunSim()
    obj_sixteen = RunSim()

    # Covers unlimited slicing single core
    t13 = threading.Thread(target=obj_thirteen.run_yue, args=(100000, 13, 1, 'first-fit', req_dist, False))
    t13.start()

    # Covers unlimited slicing four cores
    t14 = threading.Thread(target=obj_fourteen.run_yue, args=(100000, 14, 4, 'first-fit', req_dist, False))
    t14.start()

    # Covers unlimited slicing seven cores
    t15 = threading.Thread(target=obj_fifteen.run_yue, args=(100000, 15, 7, 'first-fit', req_dist, False))
    t15.start()

    # Covers unlimited slicing 10 cores
    t16 = threading.Thread(target=obj_sixteen.run_yue, args=(100000, 16, 10, 'first-fit', req_dist, False))
    t16.start()

    time.sleep(2)

    obj_seventeen = RunSim()
    obj_eighteen = RunSim()
    obj_nineteen = RunSim()
    obj_twenty = RunSim()

    # Covers 50 Gbps only single core
    req_dist = {'25': 0.0, '50': 1.0, '100': 0.0, '200': 0.0, '400': 0.0}
    t17 = threading.Thread(target=obj_seventeen.run_yue, args=(1, 17, 1, 'first-fit', req_dist, False))
    t17.start()

    t18 = threading.Thread(target=obj_eighteen.run_yue, args=(2, 18, 1, 'first-fit', req_dist, False))
    t18.start()

    t19 = threading.Thread(target=obj_nineteen.run_yue, args=(4, 19, 1, 'first-fit', req_dist, False))
    t19.start()

    t20 = threading.Thread(target=obj_twenty.run_yue, args=(8, 20, 1, 'first-fit', req_dist, False))
    t20.start()

    obj_twenty_one = RunSim()
    obj_twenty_two = RunSim()
    obj_twenty_three = RunSim()
    obj_twenty_four = RunSim()

    # Covers 50 Gbps only four cores
    t21 = threading.Thread(target=obj_twenty_one.run_yue, args=(1, 21, 4, 'first-fit', req_dist, False))
    t21.start()

    t22 = threading.Thread(target=obj_twenty_two.run_yue, args=(2, 22, 4, 'first-fit', req_dist, False))
    t22.start()

    t23 = threading.Thread(target=obj_twenty_three.run_yue, args=(4, 23, 4, 'first-fit', req_dist, False))
    t23.start()

    t24 = threading.Thread(target=obj_twenty_four.run_yue, args=(8, 24, 4, 'first-fit', req_dist, False))
    t24.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    t10.join()
    t11.join()
    t12.join()
    t13.join()
    t14.join()
    t15.join()
    t16.join()
    t17.join()
    t18.join()
    t19.join()
    t20.join()
    t21.join()
    t22.join()
    t23.join()
    t24.join()
