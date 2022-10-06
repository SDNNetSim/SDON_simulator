import math
import json

from scripts.structure_raw_data import structure_data, map_erlang_times
from scripts.engine import Engine

# TODO: Load other network (not European)
# TODO: Number of request is 1000
# TODO: Num core slots is 256


class RunSim:
    """
    Runs the simulations for this project.
    """

    def __init__(self, hold_time_mean=3600, inter_arrival_time=10, number_of_request=5000, num_iteration=1000,
                 num_core_slots=10):
        self.seed = list()
        self.hold_time_mean = hold_time_mean
        self.inter_arrival_time = inter_arrival_time
        self.number_of_request = number_of_request

        self.bw_type = {
            "100 Gbps": {
                "DP-QPSK": 3
            },
            "400 Gbps": {
                "DP-QPSK": 10
            }
        }

        self.num_iteration = num_iteration
        self.num_core_slots = num_core_slots
        self.physical_topology = {'nodes': {}, 'links': {}}
        self.link_num = 1

        self.data = structure_data()
        self.inter_arrive_dict = map_erlang_times()

        self.response = None
        self.output_file_name = None

    def save_input(self):
        """
        Saves simulation input data.
        """
        if self.output_file_name is None:
            with open('../data/input/simulation_input.json', 'w', encoding='utf-8') as file_path:
                json.dump(self.response, file_path)
        else:
            with open(f'../data/input/{self.output_file_name}', 'w', encoding='utf-8') as file_path:
                json.dump(self.response, file_path)

    def create_pt(self):
        """
        Creates the physical topology.
        """
        # This may change in the future, hence creating the same dictionary for all fibers in a link right now
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
            # Create a path from destination to source (bi-directional)
            self.physical_topology['links'][self.link_num] = {'fiber': tmp_dict, 'length': link_len, 'source': dest,
                                                              'destination': source}
            self.link_num += 1

    def create_input(self):
        """
        Creates simulation input data.
        """
        self.create_pt()
        self.response = {
            'seed': self.seed,
            'holding_time_mean': self.hold_time_mean,
            'inter_arrival_time': self.inter_arrival_time,
            'number_of_request': self.number_of_request,
            'BW_type': self.bw_type,
            'NO_iteration': self.num_iteration,
            'number_of_slot_per_core': self.num_core_slots,
            'physical_topology': self.physical_topology
        }

    def run(self):
        """
        Controls the class.
        """
        for erlang, inter_arrival_time in self.inter_arrive_dict.items():
            self.inter_arrival_time = inter_arrival_time
            self.create_input()

            self.output_file_name = f'{erlang}_erlang.json'
            self.save_input()
            self.link_num = 1

        # TODO: Multi-thread? (Give chunks of lists)
        for erlang in self.inter_arrive_dict.keys():  # pylint: disable=consider-iterating-dictionary
            if erlang == '30':
                engine = Engine(sim_input_fp=f'../data/input/{erlang}_erlang.json', erlang=erlang)
                engine.run()


if __name__ == '__main__':
    test_obj = RunSim()
    test_obj.run()
