# Standard imports
import json
import sys

# Third party imports
import networkx as nx
import numpy as np

# Project imports
from scripts.request_generator import generate
from scripts.sdn_controller import controller_main


class Engine:
    """
    Controls the SDN simulation.
    """

    def __init__(self, sim_input=None, erlang=None, sim_input_fp=None):
        self.blocking = {
            'simulations': dict(),
            'stats': dict()
        }
        self.blocking_iter = 0
        self.sim_input_fp = sim_input_fp
        self.sim_input = sim_input
        self.erlang = erlang
        self.seed = 1

        self.network_spec_db = dict()
        self.physical_topology = nx.Graph()

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
        self.blocking['stats'] = {
            'mean': self.mean,
            'variance': self.variance,
            'ci_rate': self.ci_rate,
            'ci_percent': self.ci_percent
        }

        with open(f'data/output/{self.erlang}_erlang.json', 'w', encoding='utf-8') as file_path:
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
            return

        self.mean = np.mean(block_percent_arr)
        if self.mean == 0:
            return
        self.variance = np.var(block_percent_arr)
        # Confidence interval rate
        self.ci_rate = 1.645 * (np.sqrt(self.variance) / np.sqrt(len(block_percent_arr)))
        self.ci_percent = ((2 * self.ci_rate) / np.mean(block_percent_arr)) * 100

        if self.ci_percent <= 5:
            print(f'Confidence interval of {round(self.ci_percent, 2)}% reached on simulation {simulation_number + 1}, '
                  f'ending and saving results for Erlang: {self.erlang}')
            self.save_sim_results()
            return True
        else:
            return False

    def update_blocking(self, i):
        """
        Updates the blocking dictionary based on number of iterations blocked divided by the number of requests.

        :param i: The iteration number completed
        :type i: int
        :return: None
        """
        self.blocking['simulations'][i] = self.blocking_iter / self.sim_input['number_of_request']

    def handle_arrival(self, time):
        """
        Calls the controller to handle an arrival request.

        :param time: The arrival time of the request
        :type time: float
        :return: None
        """
        rsa_res = controller_main(src=self.sorted_requests[time]["source"],
                                  dest=self.sorted_requests[time]["destination"],
                                  request_type="Arrival",
                                  physical_topology=self.physical_topology,
                                  network_spec_db=self.network_spec_db,
                                  mod_formats=self.sorted_requests[time]['mod_formats'],
                                  path=list()
                                  )

        if rsa_res is False:
            self.blocking_iter += 1
        else:
            self.requests_status.update({self.sorted_requests[time]['id']: {
                "mod_format": rsa_res[0]['mod_format'],
                "slots": rsa_res[0]['start_res_slot'],
                "path": rsa_res[0]['path']
            }})
            self.network_spec_db = rsa_res[1]
            self.physical_topology = rsa_res[2]

    def handle_release(self, time):
        """
        Calls the controller to handle a release request.

        :param time: The arrival time of the request
        :type time: float
        :return: None
        """
        if self.sorted_requests[time]['id'] in self.requests_status:
            controller_main(src=self.sorted_requests[time]["source"],
                            dest=self.sorted_requests[time]["destination"],
                            request_type="Release",
                            physical_topology=self.physical_topology,
                            network_spec_db=self.network_spec_db,
                            mod_formats=self.sorted_requests[time]['mod_formats'],
                            chosen_mod=self.requests_status[self.sorted_requests[time]['id']]['mod_format'],
                            slot_num=self.requests_status[self.sorted_requests[time]['id']]['slots'],
                            path=self.requests_status[self.sorted_requests[time]['id']]['path'],
                            )

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
                                     self.sim_input['number_of_slot_per_core']))

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

        for i in range(self.sim_input['num_iters']):
            if i == 0:
                print(f"Simulation started for Erlang: {self.erlang}.")

            self.blocking_iter = 0
            self.requests_status = dict()
            self.create_pt()

            if len(self.sim_input['seed']) == 0:
                self.seed = i + 1
            else:
                self.seed = self.sim_input['seed'][i]

            self.requests = generate(seed_no=self.seed,
                                     nodes=list(self.sim_input['physical_topology']['nodes'].keys()),
                                     holding_time_mean=self.sim_input['holding_time_mean'],
                                     inter_arrival_time_mean=self.sim_input['inter_arrival_time'],
                                     req_no=self.sim_input['number_of_request'],
                                     slot_dict=self.sim_input['bandwidth_types'])

            self.sorted_requests = dict(sorted(self.requests.items()))

            for time in self.sorted_requests:
                if self.sorted_requests[time]['request_type'] == "Arrival":
                    self.handle_arrival(time)
                elif self.sorted_requests[time]['request_type'] == "Release":
                    self.handle_release(time)

            self.update_blocking(i)
            if self.calc_blocking_stats(i):
                return

            if (i + 1) % 10 == 0 or i == 0:
                print(f'Iteration {i + 1} out of {self.sim_input["num_iters"]} completed for Erlang: {self.erlang}')
                block_percent_arr = np.array(list(self.blocking['simulations'].values()))
                print(f'Mean of blocking: {np.mean(block_percent_arr)}')
                self.save_sim_results()

        print(f'Simulation for Erlang: {self.erlang} finished.')
        self.save_sim_results()


if __name__ == '__main__':
    obj_one = Engine()
    obj_one.run()
