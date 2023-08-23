# Standard library imports
import json

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from sim_scripts.request_generator import generate
from sim_scripts.sdn_controller import SDNController
from useful_functions.handle_dirs_files import create_dir
from useful_functions.ai_functions import *  # pylint: disable=unused-wildcard-import


class Engine(SDNController):
    """
    Controls the simulation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Engine class.
        """
        self.properties = kwargs['properties']

        # Holds statistical information for each iteration in a given simulation.
        self.stats_dict = {
            'block_per_sim': dict(),
            'misc_stats': dict()
        }
        # The amount of times a type of bandwidth request was blocked
        tmp_obj = dict()
        for bandwidth in self.properties['mod_per_bw']:
            tmp_obj[bandwidth] = 0
        self.block_per_bw = tmp_obj
        # The number of requests that have been blocked in a simulation.
        self.num_blocked_reqs = 0
        # The network spectrum database
        self.net_spec_db = dict()
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
        self.num_dist_block = 0
        # Array used to take an average of the distance blocks across multiple simulation iterations
        self.dist_block_arr = np.array([])
        # Number of blocks due to a congestion constraint
        self.num_cong_block = 0
        # Array used to take an average of the congestion blocks across multiple simulation iterations
        self.cong_block_arr = np.array([])
        # A dictionary holding "snapshots" of each request. Info related to how many spectral slots are occupied,
        # active requests, and how many times this particular request was sliced
        self.request_snapshots = {}
        # Holds the request numbers of all the requests currently active in the network
        self.active_requests = set()

        # For the purposes of saving relevant simulation information to a certain pathway
        self.sim_info = f"{self.properties['network']}/{self.properties['sim_start'].split('_')[0]}/" \
                        f"{self.properties['sim_start'].split('_')[1]}"
        # Contains all methods related to artificial intelligence
        self.ai_obj = AIMethods(properties=self.properties, sim_info=self.sim_info)

        # Initialize the constructor of the SDNController class
        super().__init__(properties=self.properties, ai_obj=self.ai_obj)

    def _get_total_occupied_slots(self):
        """
        Returns the total number of occupied spectral slots and spectral slots occupied by a guard band in the network.

        :return: The total number of occupied spectral slots and guard bands in the network
        :rtype: tuple
        """
        self.active_requests = set()
        occupied_slots = 0
        guard_slots = 0

        for _, data in self.net_spec_db.items():
            for core in data['cores_matrix']:
                requests = set(core[core > 0])
                for req_num in requests:
                    self.active_requests.add(req_num)
                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        # Divide by 2 since there are identical bidirectional links
        total_occ_slots = int(occupied_slots / 2)
        total_guard_slots = int(guard_slots / 2)
        return total_occ_slots, total_guard_slots

    def _get_path_free_slots(self, path: list):
        """
        Returns the number of available spectral slots in the given path.

        :param path: The path to find available spectral slots.
        :type path: list

        :return: The total number of free spectral slots in the path.
        :rtype: int
        """
        self.active_requests = set()
        free_slots = 0

        for src, dest in zip(path, path[1:]):
            src_dest = (src, dest)
            for core in self.net_spec_db[src_dest]['cores_matrix']:
                free_slots += len(np.where(core == 0)[0])

        return free_slots

    def _save_sim_results(self):
        """
        Saves the simulation results to a file like #_erlang.json.

        :return: None
        """
        for _, obj in self.request_snapshots.items():
            for key, lst in obj.items():
                obj[key] = np.mean(lst)

        self.stats_dict['misc_stats'] = {
            'blocking_mean': self.blocking_mean,
            'blocking_variance': self.blocking_variance,
            'ci_rate_block': self.block_ci_rate,
            'ci_percent_block': self.block_ci_percent,
            'num_reqs': self.properties['num_requests'],
            'cores_per_link': self.properties['cores_per_link'],
            'hold_time_mean': self.properties['holding_time'],
            'spectral_slots': self.properties['spectral_slots'],
            'max_segments': self.properties['max_segments'],
            'trans_mean': np.mean(self.trans_arr),
            'dist_percent': np.mean(self.dist_block_arr) * 100.0,
            'cong_percent': np.mean(self.cong_block_arr) * 100.0,
            'block_per_bw': {key: np.mean(lst) for key, lst in self.block_per_bw.items()},
            'alloc_method': self.properties['allocation_method'],
            'route_method': self.properties['route_method'],
            'dynamic_lps': self.properties['dynamic_lps'],
            'is_training': self.properties['ai_arguments']['is_training'],
            'beta': self.properties['beta'],
            'request_snapshots': self.request_snapshots,
        }

        base_fp = "data/output/"

        if self.properties['ai_algorithm'] != 'None':
            self.ai_obj.save()

        # Save threads to child directories
        base_fp += f"/{self.sim_info}/{self.properties['thread_num']}"
        create_dir(base_fp)

        with open(f"{base_fp}/{self.properties['erlang']}_erlang.json", 'w', encoding='utf-8') as file_path:
            json.dump(self.stats_dict, file_path, indent=4)

    def _check_confidence_interval(self, iteration):
        """
        Checks if the confidence interval is high enough to stop the simulation.

        :param iteration: The current iteration of the simulation
        :type iteration: int

        :return: A boolean indicating whether to end the simulation or not
        """
        block_percent_arr = np.array(list(self.stats_dict['block_per_sim'].values()))
        self.blocking_mean = np.mean(block_percent_arr)
        # Cannot calculate confidence interval when given zero or only one iteration
        if self.blocking_mean == 0.0 or len(block_percent_arr) <= 1:
            return False

        self.blocking_variance = np.var(block_percent_arr)
        try:
            self.block_ci_rate = 1.645 * (np.sqrt(self.blocking_variance) / np.sqrt(len(block_percent_arr)))
            self.block_ci_percent = ((2 * self.block_ci_rate) / self.blocking_mean) * 100
        except ZeroDivisionError:
            return False

        if self.block_ci_percent <= 5:
            print(f"Confidence interval of {round(self.block_ci_percent, 2)}% reached on simulation "
                  f"{iteration + 1}, ending and saving results for Erlang: {self.properties['erlang']}")
            self._save_sim_results()
            return True

        return False

    def _calculate_block_percent(self, iteration):
        """
        Calculates the percentage of blocked requests for the current iteration and updates the blocking in the
        statistics dictionary.

        :param iteration: The iteration number completed
        :type iteration: int

        :return: None
        """
        num_requests = self.properties['num_requests']

        if num_requests == 0:
            block_percentage = 0
        else:
            block_percentage = self.num_blocked_reqs / self.properties['num_requests']

        self.stats_dict['block_per_sim'][iteration] = block_percentage

    def _handle_arrival(self, curr_time, iteration):
        """
        Updates the SDN controller to handle an arrival request. Also retrieves and calculates relevant request
        statistics.

        :param curr_time: The arrival time of the request
        :type curr_time: float

        :param iteration: The current iteration of the simulation.
        :type iteration: int

        :return: The number of transponders used for the request
        """
        request = self.reqs_dict[curr_time]
        self.req_id = request['id']
        self.source = request['source']
        self.destination = request['destination']
        self.path = None
        self.chosen_bw = request['bandwidth']

        resp = self.handle_event(request_type='arrival')

        if self.properties['ai_algorithm'] != 'None':
            if not resp[0]:
                routed = False
            else:
                routed = True
            self.ai_obj.update(routed=routed, iteration=iteration)

        # TODO: Should probably make this more readable
        # Request was blocked
        if not resp[0]:
            self.num_blocked_reqs += 1
            # No transponders used
            # Determine if blocking was due to distance or congestion
            if resp[1]:
                self.num_dist_block += 1
            else:
                self.num_cong_block += 1

            # Update how many times this bandwidth type has been blocked
            self.block_per_bw[self.chosen_bw] += 1

            # Only one transponder used (the original for the request)
            return 1

        response_data, num_transponders = resp[0], resp[2]
        self.reqs_status.update({self.req_id: {
            "mod_format": response_data['mod_format'],
            "path": response_data['path'],
            "is_sliced": response_data['is_sliced']
        }})

        self.num_trans += num_transponders

        return num_transponders

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
        self.topology.add_nodes_from(self.properties['topology_info']['nodes'])

        # Create links
        for link_num, link_data in self.properties['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']

            # Create cores matrix
            cores_matrix = self._create_cores_matrix(link_data['fiber']['num_cores'])

            # Add links to network spectrum database
            self.net_spec_db[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_db[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}

            # Add links to physical topology
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)

    def _create_cores_matrix(self, num_cores):
        """
        Creates a 2D NumPy array representing the cores matrix for a fiber link.

        :param num_cores: The number of fiber cores for the link.
        :type num_cores: int

        :return: A 2D NumPy array representing the cores matrix.
        """
        return np.zeros((num_cores, self.properties['spectral_slots']))

    def _print_iter_stats(self, iteration):
        """
        Print iteration statistics.

        :param iteration: the iteration number
        :type iteration: int

        :return: None
        """
        print(f"Iteration {iteration + 1} out of {self.properties['max_iters']} "
              f"completed for Erlang: {self.properties['erlang']}")
        block_percent_arr = np.array(list(self.stats_dict['block_per_sim'].values()))
        print(f'Mean of blocking: {np.mean(block_percent_arr)}')

    def _generate_requests(self, seed):
        """
        Generates the requests for a single iteration of the simulation.

        :param seed: The seed to use for the random number generator.
        :type seed: int

        :return: None
        """
        self.reqs_dict = generate(seed=seed,
                                  nodes=list(self.properties['topology_info']['nodes'].keys()),
                                  hold_time_mean=self.properties['holding_time'],
                                  arr_rate_mean=self.properties['arrival_rate'],
                                  num_reqs=self.properties['num_requests'],
                                  mod_per_bw=self.properties['mod_per_bw'],
                                  req_dist=self.properties['request_distribution'])

        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    def _update_transponders(self):
        """
        Updates the transponder usage array with the current transponder utilization.

        :return: None
        """
        self.trans_arr = np.append(self.trans_arr,
                                   self.num_trans / (self.properties['num_requests'] - self.num_blocked_reqs))

    def _update_blocking_distribution(self):
        """
        Updates the blocking distribution arrays with the current blocking statistics. If no requests have been blocked,
        the arrays are not updated.

        :return: None
        """
        if self.num_blocked_reqs > 0:
            self.dist_block_arr = np.append(self.dist_block_arr,
                                            float(self.num_dist_block) / float(self.num_blocked_reqs))
            self.cong_block_arr = np.append(self.cong_block_arr,
                                            float(self.num_cong_block) / float(self.num_blocked_reqs))

    def _update_request_snapshots_dict(self, request_number, num_transponders):
        """
        Updates the request snapshot dictionary with information about the current request.

        :param request_number: Represents the request number we're about to allocate
        :type request_number: int

        :param num_transponders: The number of transponders the request used
        :type num_transponders: int
        """
        occupied_slots, guard_bands = self._get_total_occupied_slots()

        self.request_snapshots[request_number]['occ_slots'].append(occupied_slots)
        self.request_snapshots[request_number]['guard_bands'].append(guard_bands)
        self.request_snapshots[request_number]['active_requests'].append(len(self.active_requests))

        blocking_prob = self.num_blocked_reqs / request_number
        self.request_snapshots[request_number]["blocking_prob"].append(blocking_prob)

        self.request_snapshots[request_number]['num_segments'].append(num_transponders)

    def _init_ai_obj(self, iteration: int):
        """
        Initializes the variables for an AI object with respect to a single iteration of the simulation.

        :param iteration: The current iteration of the simulation
        :type iteration: int
        """
        # TODO: Why not just pass properties?
        self.properties['ai_arguments']['episodes'] = self.properties['max_iters']
        self.properties['ai_arguments']['cores_per_link'] = self.properties['cores_per_link']
        self.properties['ai_arguments']['erlang'] = self.properties['erlang']
        self.properties['ai_arguments']['topology'] = self.topology
        self.properties['ai_arguments']['curr_episode'] = iteration
        self.properties['ai_arguments']['beta'] = self.properties['beta']
        self.properties['ai_arguments']['mod_per_bw'] = self.properties['mod_per_bw']

        if self.properties['ai_arguments']['is_training']:
            self.properties['ai_arguments']['table_path'] = self.sim_info

        self.ai_obj.setup(algorithm=self.properties['ai_algorithm'],
                          params=self.properties['ai_arguments'])

    def _init_iter_vars(self):
        """
        Initializes the variables for a single iteration of the simulation.

        :return: None
        """
        # Initialize variables for this iteration of the simulation
        self.num_dist_block = 0
        self.num_cong_block = 0
        self.num_trans = 0
        self.num_blocked_reqs = 0
        self.reqs_status = dict()
        for request_number in range(1, self.properties['num_requests'] + 1):
            self.request_snapshots[request_number] = {
                'occ_slots': [],
                'guard_bands': [],
                'blocking_prob': [],
                'num_segments': [],
                'active_requests': []
            }

    def run(self):
        """
        Runs the SDN simulation.

        :return: None
        """
        for iteration in range(self.properties["max_iters"]):
            self._init_iter_vars()
            self._create_topology()

            if iteration == 0:
                print(f"Simulation started for Erlang: {self.properties['erlang']} "
                      f"thread number: {self.properties['thread_num']}.")

            if self.properties['ai_algorithm'] != 'None':
                self._init_ai_obj(iteration=iteration)

            seed = self.properties["seeds"][iteration] if self.properties["seeds"] else iteration + 1
            self._generate_requests(seed)

            request_number = 1
            for curr_time in self.reqs_dict:
                req_type = self.reqs_dict[curr_time]["request_type"]
                if req_type == "arrival":
                    num_transponders = self._handle_arrival(curr_time, iteration)
                    self._update_request_snapshots_dict(request_number, num_transponders)

                    request_number += 1
                elif req_type == "release":
                    self._handle_release(curr_time)
                else:
                    raise NotImplementedError

            self._calculate_block_percent(iteration)
            self._update_blocking_distribution()
            self._update_transponders()

            # Some form of ML/RL is being used, ignore confidence intervals for training and testing
            if not self.properties['ai_algorithm'] == 'None':
                if self._check_confidence_interval(iteration):
                    return

            if (iteration + 1) % 20 == 0 or iteration == 0:
                self._print_iter_stats(iteration)

            self._save_sim_results()

        print(f"Simulation for Erlang: {self.properties['erlang']} finished.")
        self._save_sim_results()
