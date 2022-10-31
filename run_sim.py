import math
import json
import os
import threading

from scripts.structure_raw_data import structure_data, map_erlang_times
from scripts.engine import Engine


# TODO: Try one, four, and seven cores
# TODO: Change request generation to match Yue's ratios
# TODO: Also considering reachability
# TODO: Consider guard band variable


class RunSim:
    """
    Runs the simulations for this project.
    """

    def __init__(self, hold_time_mean=3600, inter_arrival_time=4.5, number_of_request=5000, num_iteration=5,
                 num_core_slots=128):
        self.seed = list()
        self.constant_hold = True
        self.hold_time_mean = hold_time_mean
        self.inter_arrival_time = inter_arrival_time
        self.number_of_request = number_of_request

        # TODO: Eventually move to a file to be read
        self.bw_type = {
            "50": {
                "QPSK": {
                    "slots_needed": 2,
                    "max_length": 11080,
                },
                "16-QAM": {
                    "slots_needed": 1,
                    "max_length": 4750,
                },
                "64-QAM": {
                    "slots_needed": 1,
                    "max_length": 1832,
                },
            },
            "100": {
                "QPSK": {
                    "slots_needed": 4,
                    "max_length": 5540,
                },
                "16-QAM": {
                    "slots_needed": 2,
                    "max_length": 2375,
                },
                "64-QAM": {
                    "slots_needed": 2,
                    "max_length": 916,
                },
            },
            "400": {
                "QPSK": {
                    "slots_needed": 16,
                    "max_length": 1385,
                },
                "16-QAM": {
                    "slots_needed": 8,
                    "max_length": 594,
                },
                "64-QAM": {
                    "slots_needed": 6,
                    "max_length": 229,
                },
            },
        }

        self.num_iteration = num_iteration
        self.num_core_slots = num_core_slots
        self.physical_topology = {'nodes': {}, 'links': {}}
        self.link_num = 1

        self.data = structure_data()
        self.hold_inter_dict = map_erlang_times()

        self.sim_input = None
        self.output_file_name = None
        self.save = True

    def save_input(self):
        """
        Saves simulation input data.
        """
        if self.output_file_name is None:
            if not os.path.exists('data/input'):
                os.mkdir('data/input')
            if not os.path.exists('data/output'):
                os.mkdir('data/output')

            with open('data/input/simulation_input.json', 'w', encoding='utf-8') as file_path:
                json.dump(self.sim_input, file_path, indent=4)
        else:
            raise NotImplementedError

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
        tmp_dict['num_cores'] = 1
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
            'BW_type': self.bw_type,
            'NO_iteration': self.num_iteration,
            'number_of_slot_per_core': self.num_core_slots,
            'physical_topology': self.physical_topology
        }

    def thread_runs(self):
        """
        Executes the run method using threads.
        """
        t1 = threading.Thread(target=self.run, args=(0, 8))
        t2 = threading.Thread(target=self.run, args=(8, 16))
        t3 = threading.Thread(target=self.run, args=(16, 24))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

    def run(self, start_erlang=None, end_erlang=None):
        """
        Controls the class.
        """
        erlang_list = list(self.hold_inter_dict.keys())

        for erlang, obj in self.hold_inter_dict.items():
            if not self.constant_hold:
                self.hold_time_mean = obj['holding_time_mean']
            # self.inter_arrival_time = obj['inter_arrival_time']
            # TODO: Eventually change this
            self.inter_arrival_time = 17.14285714285714
            self.create_input()

            if self.save:
                self.save_input()

            if erlang == '800':
                if erlang in erlang_list[start_erlang:end_erlang] or start_erlang is None:
                    engine = Engine(self.sim_input, erlang=erlang)
                    engine.run()


if __name__ == '__main__':
    test_obj = RunSim()
    # test_obj.thread_runs()
    test_obj.run()
