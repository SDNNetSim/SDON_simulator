import math
import json
import os
import threading

from scripts.structure_raw_data import structure_data, map_erlang_times
from scripts.engine import Engine


# TODO: Update docs
# TODO: Update tests
# TODO: GitHub pipelines


class RunSim:
    """
    Runs the simulations for this project.
    """

    # TODO: Increase lambda like Yue, constant Mu, calculate Erlang and save like that
    # TODO: Run and save for multiple cores iteratively
    # TODO: Output relevant data to a file like Yue?
    # TODO: Move most of this info to another file, everything here should only be running the simulation.
    def __init__(self, hold_time_mean=0.2, inter_arrival_time=2, number_of_request=30000,
                 num_iteration=100, num_core_slots=128, num_cores=1, bw_slot=12.5):
        self.seed = list()
        self.constant_hold = False
        self.number_of_request = number_of_request
        self.num_cores = num_cores
        self.hold_time_mean = hold_time_mean
        # TODO: Normalize this value?
        self.inter_arrival_time = inter_arrival_time
        # Frequency for one spectrum slot (GHz)
        self.bw_slot = bw_slot

        self.create_bw_info()
        with open('./data/input/bandwidth_info.json', 'r') as fp:
            self.bw_types = json.load(fp)

        self.num_iteration = num_iteration
        self.num_core_slots = num_core_slots
        self.physical_topology = {'nodes': {}, 'links': {}}
        self.link_num = 1

        self.data = structure_data()
        self.hold_inter_dict = map_erlang_times()

        self.sim_input = None
        self.output_file_name = None
        self.save = True

    def save_input(self, file_name=None, obj=None):
        """
        Saves simulation input data.
        """
        if not os.path.exists('data/input'):
            os.mkdir('data/input')
        if not os.path.exists('data/output'):
            os.mkdir('data/output')

        # Default to saving simulation input
        if file_name is None:
            file_name = 'simulation_input.json'
            obj = self.sim_input
        with open(f'data/input/{file_name}', 'w', encoding='utf-8') as file_path:
            json.dump(obj, file_path, indent=4)

    def create_pt(self):
        """
        Creates the physical topology.
        """
        # This may change in the future, hence creating the same dictionary for all fibers in a link right now
        # TODO: Are these exponents or 'e'?
        tmp_dict = dict()
        tmp_dict['attenuation'] = (0.2 / 4.343) * (math.e ** -3)
        tmp_dict['nonlinearity'] = 1.3 * (math.e ** -3)
        tmp_dict['dispersion'] = (16 * math.e ** -6) * ((1550 * math.e ** -9) ** 2) / (
                2 * math.pi * 3 * math.e ** 8)
        tmp_dict['num_cores'] = self.num_cores
        tmp_dict['fiber_type'] = 0

        for nodes, link_len in self.data.items():
            source = nodes[0]
            dest = nodes[1]

            self.physical_topology['nodes'][source] = {'type': 'CDC'}
            self.physical_topology['nodes'][dest] = {'type': 'CDC'}
            self.physical_topology['links'][self.link_num] = {'fiber': tmp_dict, 'length': link_len, 'source': source,
                                                              'destination': dest}
            self.link_num += 1
        # Reset link numbers
        self.link_num = 1

    # TODO: Make a config file instead
    def create_bw_info(self):
        # Max length is in km
        # TODO: (Question for Yue) Potentially change to 40 (Is this a bug?)
        bw_info = {
            '50': {'QPSK': {'max_length': 11080}, '16-QAM': {'max_length': 4750}, '64-QAM': {'max_length': 1832}},
            '100': {'QPSK': {'max_length': 5540}, '16-QAM': {'max_length': 2375}, '64-QAM': {'max_length': 916}},
            '400': {'QPSK': {'max_length': 1385}, '16-QAM': {'max_length': 594}, '64-QAM': {'max_length': 229}},
        }
        for bw, bw_obj in bw_info.items():
            for mod_format, mod_obj in bw_obj.items():
                bw_obj[mod_format]['slots_needed'] = math.ceil(float(bw) / self.bw_slot)

        self.save_input('bandwidth_info.json', bw_info)

    def create_input(self):
        """
        Creates simulation input data.
        """
        self.create_pt()
        self.sim_input = {
            'seed': self.seed,
            'holding_time_mean': self.hold_time_mean,
            'inter_arrival_time': self.inter_arrival_time,
            'number_of_request': self.number_of_request,
            'bandwidth_types': self.bw_types,
            'num_iters': self.num_iteration,
            'number_of_slot_per_core': self.num_core_slots,
            'physical_topology': self.physical_topology
        }

    def thread_runs(self):
        """
        Executes the run method using threads.
        """
        t1 = threading.Thread(target=self.run, args=(2, 48))
        t2 = threading.Thread(target=self.run, args=(48, 96))
        t3 = threading.Thread(target=self.run, args=(96, 144))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

    def run(self, lambda_start, lambda_end):
        """
        Controls the class.
        """
        for curr_lam in range(lambda_start, lambda_end, 2):
            self.inter_arrival_time = curr_lam
            self.create_input()

            if self.save:
                self.save_input()

            engine = Engine(self.sim_input, erlang=self.inter_arrival_time / self.hold_time_mean)
            engine.run()
        return


if __name__ == '__main__':
    test_obj = RunSim()
    test_obj.run(lambda_start=2, lambda_end=143)
    # test_obj.thread_runs()
