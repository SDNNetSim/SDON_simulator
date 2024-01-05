# Standard library imports
import os
import signal

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from sim_scripts.request_generator import get_requests
from sim_scripts.sdn_controller import SDNController
from useful_functions.ai_helpers import AIMethods
from useful_functions.stats_helpers import SimStats


class Engine:
    """
    Controls a single simulation.
    """

    def __init__(self, **kwargs):
        self.engine_props = kwargs['properties']

        # The network spectrum database
        self.net_spec_dict = dict()
        # Contains the requests generated in a simulation
        self.reqs_dict = None
        # Holds relevant information of requests that have been ALLOCATED in a simulation, used for debugging
        self.reqs_status_dict = dict()

        self.iteration = 0
        self.topology = nx.Graph()
        # For the purposes of saving simulation output data
        self.sim_info = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                     self.engine_props['sim_start'])

        self.sdn_obj = SDNController(properties=self.engine_props)
        self.stats_obj = SimStats(engine_props=self.engine_props, sim_info=self.sim_info)
        self.ai_obj = AIMethods(properties=self.engine_props)

    def update_ai_obj(self, sdn_data: dict):
        """
        Updates the artificial intelligent object class after each request.
        
        :param sdn_data: The data that was retrieved from the SDN controller.
        :return: None
        """
        if self.engine_props['route_method'] == 'ai':
            if not sdn_data[0]:
                routed = False
                spectrum = {}
                path_mod = ''
            else:
                spectrum = sdn_data[0]['spectrum']
                routed = True
                path_mod = sdn_data[0]['mod_format']

            self.ai_obj.update(routed=routed, spectrum=spectrum, path_mod=path_mod)

    def handle_arrival(self, curr_time: float):
        """
        Updates the SDN controller to handle an arrival request and retrieves relevant request statistics.

        :param curr_time: The arrival time of the request.
        :return: None
        """
        request_data = self.reqs_dict[curr_time]
        self.sdn_obj.req_id = request_data['id']
        self.sdn_obj.source = request_data['source']
        self.sdn_obj.destination = request_data['destination']
        self.sdn_obj.path = None
        self.sdn_obj.chosen_bw = request_data['bandwidth']
        sdn_resp = self.sdn_obj.handle_event(request_type='arrival')

        self.update_ai_obj(sdn_data=sdn_resp)
        self.stats_obj.iter_update(req_data=request_data, sdn_data=sdn_resp)

        if sdn_resp[0]:
            self.stats_obj.curr_trans = sdn_resp[2]

            self.reqs_status_dict.update({self.sdn_obj.req_id: {
                "mod_format": sdn_resp[0]['mod_format'],
                "path": sdn_resp[0]['path'],
                "is_sliced": sdn_resp[0]['is_sliced']
            }})

    def handle_release(self, curr_time: float):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request.
        :return: None
        """
        request = self.reqs_dict[curr_time]
        self.sdn_obj.req_id = request['id']
        self.sdn_obj.source = request['source']
        self.sdn_obj.destination = request['destination']
        self.sdn_obj.chosen_bw = request['bandwidth']

        if self.reqs_dict[curr_time]['id'] in self.reqs_status_dict:
            self.sdn_obj.path = self.reqs_status_dict[self.reqs_dict[curr_time]['id']]['path']
            self.sdn_obj.handle_event(request_type='release')
        # Request was blocked, nothing to release
        else:
            pass

    def create_topology(self):
        """
        Create the physical topology of the simulation.

        :return: None
        """
        self.net_spec_dict = {}
        # Create nodes
        self.topology.add_nodes_from(self.engine_props['topology_info']['nodes'])
        # Create links
        for link_num, link_data in self.engine_props['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']
            cores_matrix = np.zeros((link_data['fiber']['num_cores'], self.engine_props['spectral_slots']))

            self.net_spec_dict[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_dict[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)

        self.sdn_obj.topology = self.topology
        self.stats_obj.topology = self.topology
        self.sdn_obj.net_spec_db = self.net_spec_dict
        self.engine_props['topology'] = self.topology

    def generate_requests(self, seed: int):
        """
        Calls the request generator to generate requests.

        :param seed: The seed to use for the random generation.
        :return: None
        """
        self.reqs_dict = get_requests(seed=seed, engine_props=self.engine_props)
        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    def run(self):
        """
        Controls the Engine class methods.

        :return: None
        """
        self.create_topology()
        for iteration in range(self.engine_props["max_iters"]):
            self.iteration = iteration

            self.stats_obj.iteration = iteration
            self.stats_obj.init_iter_stats()
            signal.signal(signal.SIGINT, self.stats_obj.save_stats)
            signal.signal(signal.SIGTERM, self.stats_obj.save_stats)

            if self.engine_props['route_method'] == 'ai':
                signal.signal(signal.SIGINT, self.ai_obj.save)
                signal.signal(signal.SIGTERM, self.ai_obj.save)

                self.ai_obj.reset_epsilon()
                self.ai_obj.episode = iteration
            if iteration == 0:
                print(f"Simulation started for Erlang: {self.engine_props['erlang']} "
                      f"simulation number: {self.engine_props['thread_num']}.")

            seed = self.engine_props["seeds"][iteration] if self.engine_props["seeds"] else iteration + 1
            self.generate_requests(seed)
            req_num = 1
            for curr_time in self.reqs_dict:
                req_type = self.reqs_dict[curr_time]["request_type"]
                if req_type == "arrival":
                    self.ai_obj.req_id = req_num
                    self.handle_arrival(curr_time)

                    if req_num % self.engine_props['snapshot_step'] == 0 and self.engine_props['save_snapshots']:
                        self.stats_obj.update_snapshot(net_spec_dict=self.net_spec_dict, req_num=req_num)

                    req_num += 1
                elif req_type == "release":
                    self.handle_release(curr_time)
                else:
                    raise NotImplementedError(f'Request type unrecognized. Expected arrival or release, '
                                              f'got: {req_type}')

            self.stats_obj.get_blocking()
            self.stats_obj.end_iter_update()
            # Some form of ML/RL is being used, ignore confidence intervals for training and testing
            # TODO: What I'm going to do for now is remove ai_arguments, since I'll eventually replace it with something
            #   else
            if self.engine_props['ai_algorithm'] == 'None':
                if self.stats_obj.get_conf_inter():
                    self.ai_obj.save()
                    return

            if (iteration + 1) % self.engine_props['print_step'] == 0 or iteration == 0:
                self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'])

            self.ai_obj.save()
            self.stats_obj.save_stats()

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
