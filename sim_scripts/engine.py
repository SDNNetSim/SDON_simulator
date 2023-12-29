# Standard library imports
import json
import signal
import copy

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from sim_scripts.request_generator import generate
from sim_scripts.sdn_controller import SDNController
from useful_functions.handle_dirs_files import create_dir
from useful_functions.ai_functions import AIMethods
from useful_functions.sim_functions import find_path_len
# TODO: Add to doc standards (x_helpers.py)
from useful_functions.stats_helpers import SimStats


# TODO: Keep super for now, but change to sdn_obj and make it a standard (last)
# TODO: All changes, keep in mind: integration, integration, integration!
# TODO: Check to see if all constructor vars are used
class Engine(SDNController):
    """
    Controls the simulation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Engine class.
        """
        # TODO: Add to standards doc (x_props)
        self.engine_props = kwargs['properties']
        # TODO: Combine all stats to be saved in one object
        #   - A method or a class to handle all of this (helper functions?)
        self.stats_dict = {
            'block_per_sim': dict(),
            # TODO: Something better than 'misc_stats'
            'misc_stats': dict()
        }
        # The amount of times a type of bandwidth request was blocked
        # TODO: Better naming and way to do this
        tmp_obj = dict()
        for bandwidth in self.engine_props['mod_per_bw']:
            tmp_obj[bandwidth] = 0
        self.block_per_bw = tmp_obj
        # The number of requests that have been blocked in a simulation.
        # TODO: Here
        # self.num_blocked_reqs = 0
        # The network spectrum database
        self.net_spec_db = dict()
        self.iteration = 0
        # Used to track the total number of transponders in a simulation
        self.num_trans = 0
        # Used to take an average of the total amount of transponders for multiple simulation iterations
        self.trans_arr = np.array([])
        # Contains the requests generated in a simulation
        self.reqs_dict = None
        # Holds relevant information of requests that have been ALLOCATED in a simulation
        self.reqs_status = dict()

        # The mean of the blocking probability
        self.blocking_mean = None
        # The variance of the blocking probability
        self.blocking_variance = None
        # The confidence interval rate of the blocking probability
        self.block_ci_rate = None
        # THe confidence interval percent of the blocking probability
        self.block_ci_percent = None

        # Number of blocks due to a distance constraint
        self.block_reasons = {'distance': None, 'congestion': None, 'xt_threshold': None}
        # A dictionary holding "snapshots" of each request. Info related to how many spectral slots are occupied,
        # active requests, and how many times this particular request was sliced
        self.request_snapshots = {}
        # Holds the request numbers of all the requests currently active in the network
        self.active_requests = set()
        # TODO: Since we use super, we can move some of these to SDN controller if we want and have access
        # The number of hops for each route found
        self.hops = np.array([])
        self.cores_chosen = dict()
        # The route length (in m) for every route chosen
        self.path_lens = np.array([])
        # The time taken to route each request
        self.route_times = np.array([])
        # The weights calculated for each request, may be cross-talk, length again, or other weighted routing methods
        # TODO: Change to a dictionary
        # TODO: Shouldn't path weights be initialized after each iter?
        self.path_weights = dict()
        self.mods_used = dict()

        # For the purposes of saving relevant simulation information to a certain pathway
        # TODO: Use OS here
        self.sim_info = f"{self.engine_props['network']}/{self.engine_props['date']}/{self.engine_props['sim_start']}"

        # Initialize the constructor of the SDNController class
        super().__init__(properties=self.engine_props)
        self._create_topology()

        # TODO: Add to standards doc (x_obj)
        # TODO: Support for this later, may start up from previous statistics
        self.stats_obj = SimStats(engine_props=self.engine_props)
        # TODO: Temporary, add step and active requests to configuration file
        # TODO: Change this is done twice, each iteration
        self.stats_obj.init_stats(num_requests=self.engine_props['num_requests'], step=10,
                                  snap_keys_list=['occupied_slots', 'guard_slots', 'active_requests',
                                                  'blocking_prob', 'num_segments'],
                                  cores_range=range(self.engine_props['cores_per_link']),
                                  mod_per_bw=self.engine_props['mod_per_bw'])
        self.ai_obj = AIMethods(properties=self.engine_props)

    def _handle_arrival(self, curr_time):
        """
        Updates the SDN controller to handle an arrival request. Also retrieves and calculates relevant request
        statistics.

        :param curr_time: The arrival time of the request
        :type curr_time: float

        :return: The number of transponders used for the request
        """
        # TODO: This is the class inheritance, might want to get rid of this
        request = self.reqs_dict[curr_time]
        self.req_id = request['id']
        self.source = request['source']
        self.destination = request['destination']
        self.path = None
        self.chosen_bw = request['bandwidth']

        resp = self.handle_event(request_type='arrival')

        # TODO: Make this better, it's not good enough
        if self.engine_props['route_method'] == 'ai':
            if not resp[0]:
                routed = False
                spectrum = {}
                path_mod = ''
            else:
                spectrum = resp[0]['spectrum']
                routed = True
                path_mod = resp[0]['mod_format']
            self.ai_obj.update(routed=routed, spectrum=spectrum, path_mod=path_mod)

        # TODO: Why not just pass response data to stats helpers instead with blocked as true or false?
        # Request was blocked
        if not resp[0]:
            # TODO: Here, also update to something better than just resp or request
            self.stats_obj.get_iter_data(blocked=True, req_data=request, sdn_data=resp, topology=self.topology)
            # self.stats_obj.blocked_reqs += 1
            # self.num_blocked_reqs += 1
            # Update the reason for blocking
            # self.block_reasons[resp[1]] += 1
            # Update how many times this bandwidth type has been blocked
            # self.block_per_bw[self.chosen_bw] += 1

            # Only one transponder used (the original for the request)
            # TODO: Leave for now but will delete
            return 1

        # TODO: Here, also update to something better than just resp or request
        self.stats_obj.get_iter_data(blocked=False, req_data=request, sdn_data=resp, topology=self.topology)
        # response_data, num_transponders = resp[0], resp[2]
        #
        # # TODO: Crazy, either another file or class
        # #   - I'm thinking a class that handles all of this stuff
        # self.hops = np.append(self.hops, len(response_data['path']) - 1)
        # self.path_lens = np.append(self.path_lens, find_path_len(path=response_data['path'], topology=self.topology))
        # self.cores_chosen[response_data['spectrum']['core_num']] += 1
        # self.route_times = np.append(self.route_times, response_data['route_time'])
        # path_mod = resp[0]['mod_format']
        # self.mods_used[self.chosen_bw][path_mod] += 1
        #
        # # TODO: Maybe a method for better integration
        # if self.engine_props['check_snr'] is None or self.engine_props['check_snr'] == 'None':
        #     self.path_weights[self.chosen_bw][path_mod].append(response_data['path_weight'])
        # else:
        #     self.path_weights[self.chosen_bw][path_mod].append(response_data['xt_cost'])

        self.reqs_status.update({self.req_id: {
            "mod_format": resp[0]['mod_format'],
            "path": resp[0]['path'],
            "is_sliced": resp[0]['is_sliced']
        }})

        # self.num_trans += num_transponders

        # TODO: This is the number of transponders but it will be removed
        return resp[2]

    def _handle_release(self, curr_time):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request
        :type curr_time: float

        :return: None
        """
        request = self.reqs_dict[curr_time]
        self.req_id = request['id']
        self.source = request['source']
        self.destination = request['destination']
        self.chosen_bw = request['bandwidth']

        if self.reqs_dict[curr_time]['id'] in self.reqs_status:
            self.path = self.reqs_status[self.reqs_dict[curr_time]['id']]['path']
            self.handle_event(request_type='release')
        # Request was blocked, nothing to release
        else:
            pass

    def _create_topology(self):
        """
        Create the physical topology of the simulation.

        This method initializes the network topology and creates nodes and links based on the
        input data provided in the `sim_data` dictionary. The method also creates the cores matrix
        for each link, adds the links to the network spectrum database, and sets their length
        attribute in the physical topology.

        :return: None
        """
        self.topology = nx.Graph()
        self.net_spec_db = {}

        # Create nodes
        self.topology.add_nodes_from(self.engine_props['topology_info']['nodes'])

        # Create links
        for link_num, link_data in self.engine_props['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']

            # Create cores matrix
            cores_matrix = np.zeros((link_data['fiber']['num_cores'], self.engine_props['spectral_slots']))
            # Add links to a network spectrum database
            self.net_spec_db[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_db[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}

            # Add links to physical topology
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)
        # TODO: Change self.topology to this variable
        self.engine_props['topology'] = self.topology

    def _generate_requests(self, seed):
        """
        Generates the requests for a single iteration of the simulation.

        :param seed: The seed to use for the random number generator.
        :type seed: int

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
        Runs the SDN simulation.

        :return: None
        """
        for iteration in range(self.engine_props["max_iters"]):
            self.iteration = iteration
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
                signal.signal(signal.SIGINT, self.ai_obj.save)
                signal.signal(signal.SIGTERM, self.ai_obj.save)

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
                    num_transponders = self._handle_arrival(curr_time)

                    if request_number % 10 == 0 and self.engine_props['save_snapshots']:
                        # TODO: Here, we return number of transponders from before, but we won't do this in the future
                        self.stats_obj.get_occupied_slots(net_spec_dict=self.net_spec_db, req_num=request_number,
                                                          num_trans=num_transponders)
                        # self._update_request_snapshots_dict(request_number, num_transponders)

                    request_number += 1
                elif req_type == "release":
                    self._handle_release(curr_time)
                else:
                    raise NotImplementedError

            # self._calculate_block_percent()
            # TODO: Here
            self.stats_obj.get_blocking(num_reqs=self.engine_props['num_requests'])

            # self._update_blocking_distribution()
            # TODO: Here
            # self._update_transponders()
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
                # TODO: Here
                # self._print_iter_stats()
                self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'], iteration=iteration,
                                                erlang=self.engine_props['erlang'])

            # TODO: Here? Still need to integrate
            self.stats_obj.save_stats(iteration=iteration, sim_info=self.sim_info)
            # self._save_sim_results()

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
