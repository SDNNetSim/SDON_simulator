import math
import json

from scripts.structure_raw_data import structure_data, map_erlang_times
from scripts.engine import Engine


# TODO: Number of iterations is 1000
# TODO: Number of core slots is 256


class RunSim:
    """
    Runs the simulations for this project.
    """

    def __init__(self, hold_time_mean=3600, inter_arrival_time=10, number_of_request=5000, num_iteration=5,
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
        self.hold_time_dict = map_erlang_times()

        self.sim_input = None
        self.output_file_name = None
        self.save = False

    def save_input(self):
        """
        Saves simulation input data.
        """
        if self.output_file_name is None:
            with open('../data/input/simulation_input.json', 'w', encoding='utf-8') as file_path:
                json.dump(self.sim_input, file_path)

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

    def run(self):
        """
        Controls the class.
        """
        # TODO: Multi-thread? (Give chunks of lists)
        for erlang, hold_time in self.hold_time_dict.items():
            self.hold_time_mean = hold_time
            self.create_input()

            if not self.save:
                self.save_input()

            if erlang == '10':
                engine = Engine(self.sim_input, erlang=erlang)
                engine.run()


if __name__ == '__main__':
    test_obj = RunSim()
    test_obj.run()
