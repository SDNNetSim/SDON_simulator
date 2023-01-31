# Standard imports
import json

# Third party imports
import networkx as nx
import numpy as np

# Project imports
from sim_scripts.request_generator import generate
from sim_scripts.sdn_controller import SDNController
from useful_functions.handle_dirs_files import create_dir


class Engine:
    """
    Controls the SDN simulation.
    """

    def __init__(self, sim_input=None, erlang=None, sim_input_fp=None, network_name=None, sim_start=None,
                 assume='arash'):
        self.blocking = {
            'simulations': dict(),
            'stats': dict()
        }
        self.blocking_iter = 0
        self.assume = assume
        self.sim_input_fp = sim_input_fp
        self.sim_input = sim_input
        self.erlang = erlang
        self.seed = 1
        self.sim_start = sim_start

        self.network_name = network_name
        self.network_spec_db = dict()
        self.physical_topology = nx.Graph()

        self.control_obj = SDNController(sim_assume=assume)

        self.requests = None
        self.sorted_requests = None
        self.requests_status = dict()

        self.mean = None
        self.variance = None
        self.ci_rate = None
        self.ci_percent = None

    def save_sim_results(self):
        """
        Saves the simulation results to a file like #_erlang.json.
        """
        # We use link 1 to determine number of cores used (all links are the same at the moment)
        self.blocking['stats'] = {
            'mean': self.mean,
            'variance': self.variance,
            'ci_rate': self.ci_rate,
            'ci_percent': self.ci_percent,
            'misc_info': {
                'cores_used': self.sim_input['physical_topology']['links'][1]['fiber']['num_cores'],
                'mu': self.sim_input['mu'],
                'spectral_slots': self.sim_input['spectral_slots'],
                'max_lps': self.sim_input['max_lps']
            }
        }

        create_dir(f'data/output/{self.network_name}/{self.sim_start}/')
        with open(f'data/output/{self.network_name}/{self.sim_start}/{self.erlang}_erlang.json', 'w', encoding='utf-8') \
                as file_path:
            json.dump(self.blocking, file_path, indent=4)

    def calc_blocking_stats(self, simulation_number):
        """
        Determines if the confidence interval is high enough to stop the simulation.

        :param simulation_number: The current iteration of the simulation
        :type simulation_number: int
        :return: None
        """
        block_percent_arr = np.array(list(self.blocking['simulations'].values()))
        if len(block_percent_arr) == 1:
            return False

        self.mean = np.mean(block_percent_arr)
        if self.mean == 0:
            return False
        self.variance = np.var(block_percent_arr)
        # Confidence interval rate
        self.ci_rate = 1.645 * (np.sqrt(self.variance) / np.sqrt(len(block_percent_arr)))
        self.ci_percent = ((2 * self.ci_rate) / np.mean(block_percent_arr)) * 100

        if self.ci_percent <= 5:
            print(f'Confidence interval of {round(self.ci_percent, 2)}% reached on simulation {simulation_number + 1}, '
                  f'ending and saving results for Erlang: {self.erlang}')
            self.save_sim_results()
            return True

        return False

    def update_blocking(self, i):
        """
        Updates the blocking dictionary based on number of iterations blocked divided by the number of requests.

        :param i: The iteration number completed
        :type i: int
        :return: None
        """
        self.blocking['simulations'][i] = self.blocking_iter / self.sim_input['number_of_request']

    def update_control_obj(self, curr_time, release=False):
        """
        Updates the information for the SDN controller object initialized in the constructor.

        :param curr_time: The current time of the arrival or departure
        :type curr_time: float
        :param release: Determines if we have a path or not, an arrival request has not been assigned one yet
        :type release: bool
        :return: None
        """
        self.control_obj.network_db = self.network_spec_db
        self.control_obj.topology = self.physical_topology

        self.control_obj.req_id = self.sorted_requests[curr_time]["id"]
        self.control_obj.src = self.sorted_requests[curr_time]["source"]
        self.control_obj.dest = self.sorted_requests[curr_time]["destination"]
        # A path has not been chosen yet for an arrival
        if release:
            self.control_obj.path = self.requests_status[self.sorted_requests[curr_time]['id']]['path']
        else:
            self.control_obj.path = None

        self.control_obj.mod_formats = self.sim_input['bandwidth_types']
        self.control_obj.chosen_bw = self.sorted_requests[curr_time]['bandwidth']
        self.control_obj.max_lps = self.sim_input['max_lps']

    def handle_arrival(self, curr_time):
        """
        Calls the controller to handle an arrival request.

        :param curr_time: The arrival time of the request
        :type curr_time: float
        :return: None
        """
        self.update_control_obj(curr_time, release=False)
        resp = self.control_obj.handle_event(request_type='arrival')

        if resp is False:
            self.blocking_iter += 1
        else:
            self.requests_status.update({self.sorted_requests[curr_time]['id']: {
                "mod_format": resp[0]['mod_format'],
                "slots": resp[0]['start_slot'],
                "path": resp[0]['path'],
                "is_sliced": resp[0]['is_sliced']
            }})
            self.network_spec_db = resp[1]

    def handle_release(self, curr_time):
        """
        Calls the controller to handle a release request.

        :param curr_time: The arrival time of the request
        :type curr_time: float
        :return: None
        """
        if self.sorted_requests[curr_time]['id'] in self.requests_status:
            self.update_control_obj(curr_time, release=True)
            self.network_spec_db = self.control_obj.handle_event(request_type='release')

        # Request was blocked, nothing to release
        else:
            pass

    def create_pt(self):
        """
        Creates the physical topology for the simulation.

        :return: None
        """
        # Reset physical topology and network spectrum from previous iterations
        self.physical_topology = nx.Graph()
        self.network_spec_db = dict()

        for node in self.sim_input['physical_topology']['nodes']:
            self.physical_topology.add_node(node)

        for link_no in self.sim_input['physical_topology']['links']:
            source = self.sim_input['physical_topology']['links'][link_no]['source']
            dest = self.sim_input['physical_topology']['links'][link_no]['destination']
            cores_matrix = np.zeros((self.sim_input['physical_topology']['links']
                                     [link_no]['fiber']['num_cores'],
                                     self.sim_input['spectral_slots']))

            self.network_spec_db[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': link_no}
            self.network_spec_db[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': link_no}

            self.physical_topology.add_edge(self.sim_input['physical_topology']['links'][link_no]['source'],
                                            self.sim_input['physical_topology']['links'][link_no]['destination'],
                                            length=self.sim_input['physical_topology']['links'][link_no]['length'])

    def load_input(self):
        """
        Load and return the simulation input JSON file.
        """
        with open(self.sim_input_fp, encoding='utf-8') as json_file:
            self.sim_input = json.load(json_file)

    def run(self):
        """
        Controls the SDN simulation.

        :return: None
        """
        if self.sim_input_fp is not None:
            self.load_input()

        self.control_obj.num_cores = self.sim_input['num_cores']

        for i in range(self.sim_input['max_iters']):
            if i == 0:
                print(f"Simulation started for Erlang: {self.erlang}.")

            self.blocking_iter = 0
            self.requests_status = dict()
            self.create_pt()

            # No seed has been given, go off of the iteration number
            if len(self.sim_input['seed']) == 0:
                self.seed = i + 1
            else:
                self.seed = self.sim_input['seed'][i]

            self.requests = generate(seed_no=self.seed,
                                     nodes=list(self.sim_input['physical_topology']['nodes'].keys()),
                                     mu=self.sim_input['mu'],
                                     lam=self.sim_input['lambda'],
                                     num_requests=self.sim_input['number_of_request'],
                                     bw_dict=self.sim_input['bandwidth_types'],
                                     assume=self.assume)

            # TODO: I would first check here, also make sure request generation is the same
            self.sorted_requests = dict(sorted(self.requests.items()))

            for curr_time in self.sorted_requests:
                if self.sorted_requests[curr_time]['request_type'] == "arrival":
                    self.handle_arrival(curr_time)
                elif self.sorted_requests[curr_time]['request_type'] == "release":
                    self.handle_release(curr_time)
                else:
                    raise NotImplementedError

            self.update_blocking(i)
            # Confidence interval has been reached
            if self.calc_blocking_stats(i):
                return

            if (i + 1) % 10 == 0 or i == 0:
                print(f'Iteration {i + 1} out of {self.sim_input["max_iters"]} completed for Erlang: {self.erlang}')
                block_percent_arr = np.array(list(self.blocking['simulations'].values()))
                print(f'Mean of blocking: {np.mean(block_percent_arr)}')
                self.save_sim_results()

        print(f'Simulation for Erlang: {self.erlang} finished.')
        self.save_sim_results()


if __name__ == '__main__':
    obj_one = Engine()
    obj_one.run()
