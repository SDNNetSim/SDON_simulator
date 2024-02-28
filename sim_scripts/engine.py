# Standard library imports
import os
import signal

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from sim_scripts.request_generator import get_requests
from sim_scripts.sdn_controller import SDNController
from helper_scripts.ai_helpers import AIMethods
from helper_scripts.stats_helpers import SimStats


# TODO: No support for AI in all scripts for the time being
# TODO: Update docstrings
class Engine:
    """
    Controls a single simulation.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props

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

        self.sdn_obj = SDNController(engine_props=self.engine_props)
        self.stats_obj = SimStats(engine_props=self.engine_props, sim_info=self.sim_info)
        self.ai_obj = AIMethods(engine_props=self.engine_props)

    def update_ai_obj(self):
        """
        Updates the artificial intelligent object class after each request.
        """
        if self.engine_props['ai_algorithm'] is not None and self.engine_props['ai_algorithm'] != 'None':
            self.ai_obj.update(was_routed=self.sdn_obj.sdn_props['was_routed'])

    # TODO: Curr time to constructor
    def update_arrival_params(self, curr_time: float):
        # TODO: Remove or update (This needs to move to another function)
        # self.update_ai_obj()

        # TODO: Change this to not handle sdn data! Generalize
        self.stats_obj.iter_update(req_data=self.reqs_dict[curr_time], sdn_data=self.sdn_obj.sdn_props)

        # TODO: Will not use sdn props anymore
        if self.sdn_obj.sdn_props['was_routed']:
            self.stats_obj.curr_trans = self.sdn_obj.sdn_props['num_trans']

            # TODO: Will not use sdn props anymore, will have an option
            self.reqs_status_dict.update({self.reqs_dict[curr_time]['req_id']: {
                "mod_format": self.sdn_obj.sdn_props['spectrum_dict']['modulation'],
                "path": self.sdn_obj.sdn_props['path_list'],
                "is_sliced": self.sdn_obj.sdn_props['is_sliced'],
                "was_routed": self.sdn_obj.sdn_props['was_routed'],
            }})

    def handle_arrival(self, curr_time: float, ai_flag: bool):
        """
        Updates the SDN controller to handle an arrival request and retrieves relevant request statistics.

        :param curr_time: The arrival time of the request.
        """
        for req_key, req_value in self.reqs_dict[curr_time].items():
            self.sdn_obj.sdn_props[req_key] = req_value

        if not ai_flag:
            self.sdn_obj.handle_event(request_type='arrival')
            self.net_spec_dict = self.sdn_obj.sdn_props['net_spec_dict']

        return self.net_spec_dict

    def handle_release(self, curr_time: float):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request.
        """
        for req_key, req_value in self.reqs_dict[curr_time].items():
            self.sdn_obj.sdn_props[req_key] = req_value

        if self.reqs_dict[curr_time]['req_id'] in self.reqs_status_dict:
            self.sdn_obj.sdn_props['path_list'] = self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']]['path']
            self.sdn_obj.handle_event(request_type='release')
            self.net_spec_dict = self.sdn_obj.sdn_props['net_spec_dict']
        # Request was blocked, nothing to release
        else:
            pass

    def create_topology(self):
        """
        Create the physical topology of the simulation.
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

        self.engine_props['topology'] = self.topology
        self.stats_obj.topology = self.topology
        self.sdn_obj.sdn_props['net_spec_dict'] = self.net_spec_dict
        self.sdn_obj.sdn_props['topology'] = self.topology

        self.ai_obj.setup()
        self.sdn_obj.ai_obj = self.ai_obj

    def generate_requests(self, seed: int):
        """
        Calls the request generator to generate requests.

        :param seed: The seed to use for the random generation.
        """
        self.reqs_dict = get_requests(seed=seed, engine_props=self.engine_props)
        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    # TODO: ai flag to constructor
    def handle_request(self, curr_time: float, req_num: int, ai_flag: bool):
        req_type = self.reqs_dict[curr_time]["request_type"]
        if req_type == "arrival":
            self.ai_obj.req_id = req_num
            self.handle_arrival(curr_time=curr_time, ai_flag=ai_flag)
            self.update_arrival_params(curr_time=curr_time)

            if self.engine_props['save_snapshots'] and req_num % self.engine_props['snapshot_step'] == 0:
                self.stats_obj.update_snapshot(net_spec_dict=self.net_spec_dict, req_num=req_num)

            req_num += 1
        elif req_type == "release":
            # TODO: This will need a similar structure to AI flag
            self.handle_release(curr_time=curr_time)
        else:
            raise NotImplementedError(f'Request type unrecognized. Expected arrival or release, '
                                      f'got: {req_type}')

    # TODO: Iteration to constructor
    def end_iter(self, iteration: int):
        # TODO: Move this to end iter
        self.stats_obj.get_blocking()
        self.stats_obj.end_iter_update()
        # Some form of ML/RL is being used, ignore confidence intervals for training and testing
        if self.engine_props['ai_algorithm'] == 'None' or self.engine_props['ai_algorithm'] is None:
            if self.stats_obj.get_conf_inter():
                self.ai_obj.save()
                return

        if (iteration + 1) % self.engine_props['print_step'] == 0 or iteration == 0:
            self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'])

        self.ai_obj.save()
        self.stats_obj.save_stats()

    # TODO: May have a problem with iteration here
    def init_iter(self, iteration: int):
        self.iteration = iteration

        self.stats_obj.iteration = iteration
        self.stats_obj.init_iter_stats()
        # To prevent incomplete saves
        signal.signal(signal.SIGINT, self.stats_obj.save_stats)
        signal.signal(signal.SIGTERM, self.stats_obj.save_stats)

        if self.engine_props['ai_algorithm'] is not None and self.engine_props['ai_algorithm'] != 'None':
            signal.signal(signal.SIGINT, self.ai_obj.save)
            signal.signal(signal.SIGTERM, self.ai_obj.save)

            self.ai_obj.reset_epsilon()
            self.ai_obj.episode = iteration
        if iteration == 0:
            print(f"Simulation started for Erlang: {self.engine_props['erlang']} "
                  f"simulation number: {self.engine_props['thread_num']}.")

        seed = self.engine_props["seeds"][iteration] if self.engine_props["seeds"] else iteration + 1
        self.generate_requests(seed)

    def run(self):
        """
        Controls the Engine class methods.
        """
        self.create_topology()
        for iteration in range(self.engine_props["max_iters"]):
            self.init_iter(iteration=iteration)
            req_num = 1
            for curr_time in self.reqs_dict:
                # TODO: This needs to return
                self.handle_request(curr_time=curr_time, req_num=req_num, ai_flag=False)

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
