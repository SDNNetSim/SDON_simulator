# Standard library imports
import os

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from sim_scripts.request_generator import generate
from sim_scripts.sdn_controller import SDNController
from useful_functions.ai_functions import AIMethods
from useful_functions.stats_helpers import SimStats


# TODO: Keep super for now, but change to sdn_obj and make it a standard (last)
class Engine(SDNController):
    """
    Controls a single simulation.
    """

    def __init__(self, **kwargs):
        self.engine_props = kwargs['properties']
        
        # The network spectrum database
        self.net_spec_dict = dict()
        # Contains the requests generated in a simulation
        self.reqs_dict = None
        # Holds relevant information of requests that have been ALLOCATED in a simulation
        self.reqs_status_dict = dict()
        
        self.iteration = 0
        # For the purposes of saving relevant simulation information to a certain pathway
        self.sim_info = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                     self.engine_props['sim_start'])

        # Initialize the constructor of the SDNController class
        super().__init__(properties=self.engine_props)
        self._create_topology()
        # TODO: Change name of engine props
        self.stats_obj = SimStats(engine_props=self.engine_props)
        self.ai_obj = AIMethods(properties=self.engine_props)

    def _handle_arrival(self, curr_time: float):
        """
        Updates the SDN controller to handle an arrival request and retrieves relevant request statistics.

        :param curr_time: The arrival time of the request.
        :return: None
        """
        # TODO: This is the class inheritance, might want to get rid of this
        request = self.reqs_dict[curr_time]
        self.req_id = request['id']
        self.source = request['source']
        self.destination = request['destination']
        self.path = None
        self.chosen_bw = request['bandwidth']
        sdn_resp = self.handle_event(request_type='arrival')

        # TODO: Make this better, it's not good enough
        if self.engine_props['route_method'] == 'ai':
            if not sdn_resp[0]:
                routed = False
                spectrum = {}
                path_mod = ''
            else:
                spectrum = sdn_resp[0]['spectrum']
                routed = True
                path_mod = sdn_resp[0]['mod_format']
            self.ai_obj.update(routed=routed, spectrum=spectrum, path_mod=path_mod)

        # TODO: Using resp twice here
        self.stats_obj.get_iter_data(resp=sdn_resp, req_data=request, sdn_data=sdn_resp, topology=self.topology)

        if sdn_resp[0]:
            self.reqs_status_dict.update({self.req_id: {
                "mod_format": sdn_resp[0]['mod_format'],
                "path": sdn_resp[0]['path'],
                "is_sliced": sdn_resp[0]['is_sliced']
            }})

    def _handle_release(self, curr_time: float):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request.
        :return: None
        """
        request = self.reqs_dict[curr_time]
        self.req_id = request['id']
        self.source = request['source']
        self.destination = request['destination']
        self.chosen_bw = request['bandwidth']

        if self.reqs_dict[curr_time]['id'] in self.reqs_status_dict:
            self.path = self.reqs_status_dict[self.reqs_dict[curr_time]['id']]['path']
            self.handle_event(request_type='release')
        # Request was blocked, nothing to release
        else:
            pass

    def _create_topology(self):
        """
        Create the physical topology of the simulation.

        :return: None
        """
        self.topology = nx.Graph()
        self.net_spec_dict = {}

        # Create nodes
        self.topology.add_nodes_from(self.engine_props['topology_info']['nodes'])
        # Create links
        for link_num, link_data in self.engine_props['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']

            # Create cores matrix
            cores_matrix = np.zeros((link_data['fiber']['num_cores'], self.engine_props['spectral_slots']))
            # Add links to a network spectrum database
            self.net_spec_dict[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_dict[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}

            # Add links to physical topology
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)
        # TODO: Change self.topology to this variable
        self.engine_props['topology'] = self.topology

    def _generate_requests(self, seed: int):
        """
        Generates the requests for the simulation.

        :param seed: The seed to use for the random number generator.
        :return: None
        """
        # TODO: Crazy, too many params, send props
        self.reqs_dict = generate(seed=seed,
                                  nodes=list(self.engine_props['topology_info']['nodes'].keys()),
                                  hold_time_mean=self.engine_props['holding_time'],
                                  arr_rate_mean=self.engine_props['arrival_rate'],
                                  num_reqs=self.engine_props['num_requests'],
                                  mod_per_bw=self.engine_props['mod_per_bw'],
                                  req_dist=self.engine_props['request_distribution'],
                                  sim_type=self.engine_props['sim_type'])

        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    def run(self):
        """
        Runs the Engine's methods.

        :return: None
        """
        for iteration in range(self.engine_props["max_iters"]):
            self.iteration = iteration
            # TODO: Step encoded to 10 temporarily
            self.stats_obj.init_stats(num_requests=self.engine_props['num_requests'], step=10,
                                      snap_keys_list=['occupied_slots', 'guard_slots', 'active_requests',
                                                      'blocking_prob', 'num_segments'],
                                      cores_range=range(self.engine_props['cores_per_link']),
                                      mod_per_bw=self.engine_props['mod_per_bw'])

            # TODO: Add this to stats_helpers.py
            # signal.signal(signal.SIGINT, self.stats_obj.save_stats)
            # signal.signal(signal.SIGTERM, self._save_sim_results)

            if self.engine_props['route_method'] == 'ai':
                self.ai_obj.reset_epsilon()
                self.ai_obj.episode = iteration
                # TODO: Not sure what to do for integration here
                # signal.signal(signal.SIGINT, self.ai_obj.save)
                # signal.signal(signal.SIGTERM, self.ai_obj.save)

            if iteration == 0:
                print(f"Simulation started for Erlang: {self.engine_props['erlang']} "
                      f"simulation number: {self.engine_props['thread_num']}.")

            seed = self.engine_props["seeds"][iteration] if self.engine_props["seeds"] else iteration + 1
            self._generate_requests(seed)

            request_number = 1
            for curr_time in self.reqs_dict:
                req_type = self.reqs_dict[curr_time]["request_type"]
                if req_type == "arrival":
                    self.ai_obj.req_id = request_number
                    self._handle_arrival(curr_time)

                    # TODO: Config file for this, request_number % 10
                    if request_number % 10 == 0 and self.engine_props['save_snapshots']:
                        self.stats_obj.get_occupied_slots(net_spec_dict=self.net_spec_dict, req_num=request_number)

                    request_number += 1
                elif req_type == "release":
                    self._handle_release(curr_time)
                else:
                    raise NotImplementedError(f'Request type unrecongnized. Expected arrival or release, '
                                              f'got: {req_type}')

            self.stats_obj.get_blocking(num_reqs=self.engine_props['num_requests'])
            self.stats_obj.end_iter_stats(num_reqs=self.engine_props['num_requests'])
            # Some form of ML/RL is being used, ignore confidence intervals for training and testing
            # TODO: What? Clean this up
            if self.engine_props['ai_algorithm'] == 'None':
                # TODO: Here
                # if self._check_confidence_interval():
                if self.stats_obj.get_conf_inter(iteration=iteration, erlang=self.engine_props['erlang']):
                    return
            else:
                if not self.engine_props['ai_arguments']['is_training']:
                    # TODO: Here
                    # if self._check_confidence_interval():
                    if self.stats_obj.get_conf_inter(iteration=iteration, erlang=self.engine_props['erlang']):
                        return

            # TODO: Add to config file, the amount of times we want to print
            if (iteration + 1) % 20 == 0 or iteration == 0:
                self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'], iteration=iteration,
                                                erlang=self.engine_props['erlang'])

            self.stats_obj.save_stats(iteration=iteration, sim_info=self.sim_info)

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
