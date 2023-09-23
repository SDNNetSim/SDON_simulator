# Standard library imports
import json
import time
from datetime import datetime
import copy

# Third-party library imports
import concurrent.futures

# Local application imports
from handle_data.structure_data import create_network
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir
from config.setup_config import read_config
from config.parse_args import parse_args


# TODO: Structure this so objects are not re-created every time :D
# TODO: Update tests
# TODO: Update docs


class NetworkSimulator:
    """
    Controls all simulations for this project.
    """

    def __init__(self):
        """
        Initializes the NetworkSimulator class.
        """
        # Contains all the desired network simulator parameters for every thread
        self.properties = None

    def save_input(self, file_name: str = None, data: dict = None):
        """
        Saves simulation input data. Does not save bandwidth data, as that is intended to be a constant and unchanged
        file.

        :param file_name: The name of the file to save the input data to.
        :type file_name: str

        :param data: The data to save to the file.
        :type data: dict

        :return: None
        """
        path = f"data/input/{self.properties['network']}/{self.properties['date']}/{self.properties['sim_start']}"

        create_dir(path)
        create_dir('data/output')

        with open(f"{path}/{file_name}", 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def create_input(self, local_props):
        """
        Create the input data for the simulation.

        The method generates bandwidth information, creates the physical topology of the network,
        and creates a dictionary containing all the necessary simulation parameters.

        :return: None
        """
        bw_info = create_bw_info(sim_type=local_props['sim_type'])
        bw_file = f"bw_info_{local_props['thread_num']}.json"

        self.save_input(file_name=bw_file, data=bw_info)

        path = f"./data/input/{local_props['network']}/{local_props['date']}/" \
               f"{local_props['sim_start']}/{bw_file}"
        with open(path, 'r', encoding='utf-8') as file_object:
            local_props['mod_per_bw'] = json.load(file_object)

        network_data = create_network(const_weight=local_props['const_link_weight'],
                                      net_name=local_props['network'])
        local_props['topology_info'] = create_pt(cores_per_link=local_props['cores_per_link'],
                                                 network_data=network_data)

        return local_props

    def _run_yue(self, arr_rate_mean, start):
        arr_rate_mean = float(arr_rate_mean)
        local_props = copy.deepcopy(self.properties)

        local_props['erlang'] = arr_rate_mean / local_props['holding_time']
        arr_rate_mean *= float(local_props['cores_per_link'])
        local_props['arrival_rate'] = arr_rate_mean
        self.create_input(local_props=local_props)

        if arr_rate_mean == start:
            file_name = f"sim_input_{self.properties['thread_num']}.json"
            self.save_input(file_name=file_name, data=local_props)

        engine = Engine(properties=local_props)
        engine.run()

    def run_yue(self):
        """
        Runs a simulation using the Yue's simulation assumptions. Reference: Wang, Yue. Dynamic Traffic Scheduling
        Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss. University of Massachusetts Lowell, 2022.

        :return: None
        """
        arr_rate_obj = self.properties['arrival_rate']
        start, stop, step = arr_rate_obj['start'], arr_rate_obj['stop'], arr_rate_obj['step']

        if self.properties['thread_erlangs']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for arr_rate_mean in range(start, stop, step):
                    time.sleep(1.0)
                    future = executor.submit(self._run_yue, arr_rate_mean=arr_rate_mean, start=start)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for arr_rate_mean in range(start, stop, step):
                self._run_yue(arr_rate_mean=arr_rate_mean, start=start)

    def _run_arash(self, erlang, first_erlang):
        local_props = copy.deepcopy(self.properties)
        local_props['arrival_rate'] = (local_props['cores_per_link'] * erlang) / local_props['holding_time']
        local_props['erlang'] = erlang

        local_props = self.create_input(local_props=local_props)

        if first_erlang:
            self.save_input(file_name=f"sim_input_{local_props['thread_num']}.json", data=local_props)

        engine = Engine(properties=local_props)
        engine.run()

    def run_arash(self):
        """
        Runs a simulation using the Arash's simulation assumptions.
        Reference: https://doi.org/10.1016/j.comnet.2020.107755.

        :return: None
        """
        erlang_obj = self.properties['erlangs']
        start, stop, step = erlang_obj['start'], erlang_obj['stop'], erlang_obj['step']
        erlang_lst = [float(erlang) for erlang in range(start, stop, step)]

        if self.properties['thread_erlangs']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for erlang in erlang_lst:
                    first_erlang = erlang == erlang_lst[0]
                    time.sleep(1.0)
                    future = executor.submit(self._run_arash, erlang=erlang, first_erlang=first_erlang)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for erlang in erlang_lst:
                first_erlang = erlang == erlang_lst[0]
                self._run_arash(erlang=erlang, first_erlang=first_erlang)

    def run_sim(self, **kwargs):
        """
        Controls the networking simulator class.

        :return: None
        """
        self.properties = kwargs['thread_params']
        # The date and current time derived from the simulation start
        self.properties['date'] = kwargs['sim_start'].split('_')[0]

        tmp_list = kwargs['sim_start'].split('_')
        time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
        self.properties['sim_start'] = time_string

        # To keep track of each thread run and save results
        self.properties['thread_num'] = kwargs['thread_num']

        if self.properties['sim_type'] == 'yue':
            self.run_yue()

        self.run_arash()


def run(threads_obj: dict):
    """
    Runs multiple simulations concurrently using threads.

    :param threads_obj: Dictionaries, where each contains the parameters for a single thread.
    :type threads_obj: dict

    :return: None
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        for thread_num, thread_params in threads_obj.items():
            sim_obj = NetworkSimulator()
            class_inst = sim_obj.run_sim

            time.sleep(1.0)
            future = executor.submit(class_inst, thread_num=thread_num, thread_params=thread_params,
                                     sim_start=sim_start)

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == '__main__':
    args_obj = parse_args()
    threads_obj = read_config(args_obj=args_obj)
    run(threads_obj=threads_obj)
