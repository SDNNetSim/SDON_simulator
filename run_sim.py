import json
import os
import time

from scripts.structure_data import structure_data
from scripts.generate_data import create_bw_info, create_pt
from scripts.engine import Engine


# TODO: Update docs
# TODO: Update tests
# TODO: Objectify everything?


class RunSim:
    """
    Runs the simulations for this project.
    """

    def __init__(self, mu=1.0, lam=2.0, num_requests=10000, max_iters=1, spectral_slots=256, num_cores=1,  # pylint: disable=invalid-name
                 bw_slot=12.5, sim_flag='arash', constant_weight=True):

        # Assumptions for things like mu, lambda, modulation format/calc, and routing
        self.sim_flag = sim_flag
        self.network_name = None
        self.constant_weight = constant_weight
        self.seed = list()
        self.num_requests = num_requests
        self.num_cores = num_cores
        self.mu = mu  # pylint: disable=invalid-name
        self.lam = lam

        self.sim_start = time.strftime("%m%d_%H:%M:%S")

        # Frequency for one spectrum slot (GHz)
        self.bw_slot = bw_slot
        self.bw_types = None
        self.spectral_slots = spectral_slots
        self.link_num = 1

        # If the confidence interval isn't reached, maximum allowed iterations
        self.max_iters = max_iters
        self.sim_input = None
        self.output_file_name = None
        self.save = True

    def save_input(self, file_name=None, obj=None):
        """
        Saves simulation input data. Not bandwidth data for now, since that is intended to be a constant and unchanged
        file.
        """
        if not os.path.exists('data/input'):
            os.mkdir('data/input')
        if not os.path.exists('data/output'):
            os.mkdir('data/output')

        # Default file name for saving simulation input
        if file_name is None:
            file_name = 'simulation_input.json'
            obj = self.sim_input

        with open(f'data/input/{file_name}', 'w', encoding='utf-8') as file_path:
            json.dump(obj, file_path, indent=4)

    def create_input(self):
        """
        Creates simulation input data.
        """
        bw_info = create_bw_info(assume=self.sim_flag)
        self.save_input('bandwidth_info.json', bw_info)
        with open('./data/input/bandwidth_info.json', 'r', encoding='utf-8') as fp:  # pylint: disable=invalid-name
            self.bw_types = json.load(fp)

        data = structure_data(constant_weight=self.constant_weight, network=self.network_name)
        physical_topology = create_pt(num_cores=self.num_cores, nodes_links=data)

        self.sim_input = {
            'seed': self.seed,
            'mu': self.mu,
            'lambda': self.lam,
            'number_of_request': self.num_requests,
            'bandwidth_types': self.bw_types,
            'max_iters': self.max_iters,
            'spectral_slots': self.spectral_slots,
            'physical_topology': physical_topology
        }

    def thread_runs(self):
        """
        Executes the run methods using threads.
        """
        raise NotImplementedError

    def run_yue(self):
        """
        Run the simulator based on Yue Wang's previous research assumptions regarding:
            - The number of spectral slots per core
            - Slots needed for modulation formats and maximum reach
            - Link weights in the topology
            - The value of mu

        :return: None
        """
        self.mu = 0.2
        self.spectral_slots = 128
        self.sim_flag = 'yue'
        self.network_name = 'USNet'
        self.constant_weight = False

        for lam in range(2, 143, 2):
            curr_erlang = float(lam) / self.mu
            lam *= float(self.num_cores)
            self.lam = float(lam)
            self.create_input()

            if self.save:
                self.save_input()

            engine = Engine(self.sim_input, erlang=curr_erlang, network_name=self.network_name,
                            sim_start=self.sim_start,
                            assume=self.sim_flag)
            engine.run()

    def run_arash(self):
        """
        Run the simulator based on Arash Rezaee's previous research assumptions regarding:
            - The number of spectral slots per core
            - Slots needed for modulation formats
            - Link weights in the topology
            - The value of mu

        :return: None
        """
        self.mu = 3600.0
        self.spectral_slots = 256
        self.sim_flag = 'arash'
        self.network_name = 'Pan-European'
        self.constant_weight = True
        erlang_lst = [float(erlang) for erlang in range(50, 850, 50)]

        for curr_erlang in erlang_lst:
            self.lam = self.mu * float(self.num_cores) * curr_erlang
            self.create_input()

            if self.save:
                self.save_input()

            engine = Engine(self.sim_input, erlang=curr_erlang, network_name=self.network_name,
                            sim_start=self.sim_start,
                            assume=self.sim_flag)
            engine.run()


if __name__ == '__main__':
    test_obj = RunSim()
    # test_obj.run_yue()
    test_obj.run_arash()
