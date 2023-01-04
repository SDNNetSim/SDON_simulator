import math
import json
import os
import threading

from scripts.structure_raw_data import structure_data
from scripts.engine import Engine


# TODO: Update docs
# TODO: Update tests
# TODO: GitHub pipelines
# TODO: Save simulation results by topology and date directory
# TODO: Document mu and the number of cores used


class RunSim:
    """
    Runs the simulations for this project.
    """
    def __init__(self, mu=1.0, lam=2.0, num_requests=10000, max_iters=1, spectral_slots=256, num_cores=1, bw_slot=12.5):
        self.seed = list()
        self.num_requests = num_requests
        self.num_cores = num_cores
        self.mu = mu
        self.lam = lam
        self.erlang_lst = [float(erlang) for erlang in range(50, 850, 50)]

        # TODO: Update this
        # TODO: Change to handle data, structure and generate
        self.create_bw_info()
        with open('./data/input/bandwidth_info.json', 'r') as fp:
            self.bw_types = json.load(fp)
        self.data = structure_data()

        # Frequency for one spectrum slot (GHz)
        self.bw_slot = bw_slot
        self.spectral_slots = spectral_slots
        self.physical_topology = {'nodes': {}, 'links': {}}
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

    # TODO: Move this method to a "generate data" class or something
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
        # TODO: Separate this is a more organized way (Yue vs. Arash's)
        bw_info = {
            # '50': {'QPSK': {'max_length': 11080}, '16-QAM': {'max_length': 4750}, '64-QAM': {'max_length': 1832}},
            '100': {'QPSK': {'max_length': 5540}, '16-QAM': {'max_length': 2375}, '64-QAM': {'max_length': 916}},
            '400': {'QPSK': {'max_length': 1385}, '16-QAM': {'max_length': 594}, '64-QAM': {'max_length': 229}},
        }
        for bw, bw_obj in bw_info.items():
            for mod_format, mod_obj in bw_obj.items():
                if bw == '100':
                    bw_obj[mod_format]['slots_needed'] = 3
                elif bw == '400':
                    bw_obj[mod_format]['slots_needed'] = 10
                # bw_obj[mod_format]['slots_needed'] = math.ceil(float(bw) / self.bw_slot)

        self.save_input('bandwidth_info.json', bw_info)

    def create_input(self):
        """
        Creates simulation input data.
        """
        self.create_pt()
        self.sim_input = {
            'seed': self.seed,
            'mu': self.mu,
            'lambda': self.lam,
            'number_of_request': self.num_requests,
            'bandwidth_types': self.bw_types,
            'max_iters': self.max_iters,
            'spectral_slots': self.spectral_slots,
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

    def run(self):
        """
        Controls the class.
        """
        for curr_erlang in self.erlang_lst:
            self.lam = self.mu * float(self.num_cores) * curr_erlang
            self.create_input()

            # Save simulation input, if desired
            if self.save:
                self.save_input()

            engine = Engine(self.sim_input, erlang=curr_erlang)
            engine.run()
        return


if __name__ == '__main__':
    test_obj = RunSim()
    test_obj.run()
