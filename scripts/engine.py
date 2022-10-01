# Third party imports
import networkx as nx
import numpy as np

# Project imports
from scripts.request_generator import generate
from scripts.load_input import load_input
from scripts.sdn_controller import controller_main


# TODO: This will break on multi-hop routing!


class Engine:
    """
    Controls the SDN simulation.
    """

    def __init__(self):
        self.blocking = dict()
        self.blocking_iter = 0
        self.sim_input = None

        self.network_spec_db = dict()
        self.physical_topology = nx.Graph()

        self.requests = None
        self.sorted_requests = None
        self.requests_status = dict()

    def update_blocking(self, i):
        """
        Updates the blocking dictionary based on number of iterations blocked divided by the number of requests.

        :param i: The iteration number completed
        :type i: int
        :return: None
        """
        self.blocking.update({i: self.blocking_iter / self.sim_input['number_of_request']})

    def handle_arrival(self, time):
        """
        Calls the controller to handle an arrival request.

        :param time: The arrival time of the request
        :type time: float
        :return: None
        """
        rsa_res = controller_main(src=self.sorted_requests[time]["source"][0],
                                  dest=self.sorted_requests[time]["destination"][0],
                                  request_type="Arrival",
                                  physical_topology=self.physical_topology,
                                  network_spec_db=self.network_spec_db,
                                  num_slots=self.sorted_requests[time]['number_of_slot'][0],
                                  slot_num=-1,
                                  path=list()
                                  )
        if rsa_res is False:
            self.blocking_iter += 1
        else:
            self.requests_status.update({self.sorted_requests[time]['id']: {
                "slots": rsa_res[0]['starting_NO_reserved_slot'],
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
            controller_main(src=self.sorted_requests[time]["source"][0],
                            dest=self.sorted_requests[time]["destination"][0],
                            request_type="Release",
                            physical_topology=self.physical_topology,
                            network_spec_db=self.network_spec_db,
                            num_slots=self.sorted_requests[time]['number_of_slot'][0],
                            slot_num=self.requests_status[self.sorted_requests[time]['id']]['slots'],
                            path=self.requests_status[self.sorted_requests[time]['id']]['path'],
                            )

    def create_pt(self):
        """
        Creates the physical topology for the simulation.

        :return: None
        """
        for node in self.sim_input['physical_topology']['nodes']:
            self.physical_topology.add_node(node)

        for link_no in self.sim_input['physical_topology']['links']:
            self.network_spec_db.update({(self.sim_input['physical_topology']['links'][link_no]['source'],
                                          self.sim_input['physical_topology']['links'][link_no]['destination']):
                                             np.zeros((self.sim_input['physical_topology']['links']
                                                       [link_no]['fiber']['num_cores'],
                                                       self.sim_input['number_of_slot_per_lisnk']))})

            self.physical_topology.add_edge(self.sim_input['physical_topology']['links'][link_no]['source'],
                                            self.sim_input['physical_topology']['links'][link_no]['destination'],
                                            length=self.sim_input['physical_topology']['links'][link_no]['length'])

    def load_input(self):
        """
        Loads the simulation JSON file.

        :return: None
        """
        self.sim_input = load_input()

    def run(self):
        """
        Controls the SDN simulation.

        :return: None
        """
        self.load_input()

        for i in range(self.sim_input['NO_iteration']):
            self.blocking_iter = 0
            self.requests_status = dict()
            self.create_pt()

            self.requests = generate(seed_no=self.sim_input['seed'],
                                     nodes=list(self.sim_input['physical_topology']['nodes'].keys()),
                                     holding_time_mean=self.sim_input['holding_time_mean'],
                                     inter_arrival_time_mean=self.sim_input['inter_arrival_time'],
                                     req_no=self.sim_input['number_of_request'],
                                     slot_list=self.sim_input['BW_type'])

            self.sorted_requests = dict(sorted(self.requests.items()))

            for time in self.sorted_requests:
                if self.sorted_requests[time]['request_type'] == "Arrival":
                    self.handle_arrival(time)
                elif self.sorted_requests[time]['request_type'] == "Release":
                    self.handle_release(time)

            self.update_blocking(i)


if __name__ == '__main__':
    obj_one = Engine()
    obj_one.run()
