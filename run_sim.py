# Standard library imports
import json
import time

# Third-party library imports
import concurrent.futures

# Local application imports
from handle_data.structure_data import create_network
from handle_data.generate_data import create_bw_info, create_pt
from sim_scripts.engine import Engine
from useful_functions.handle_dirs_files import create_dir
from config.setup_config import read_config


# TODO: Structure this so objects are not re-created every time :D
#   - This would also add the support of threading different traffic volumes
# TODO: Update tests
# TODO: Update docs
# TODO: Multiple class inheritance instead of updating local constructors over and over?


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
        path = f"data/input/{self.properties['network']}/{self.properties['date']}/{self.properties['curr_time']}"

        create_dir(path)
        create_dir('data/output')

        with open(f"{path}/{file_name}", 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def create_input(self):
        """
        Create the input data for the simulation.

        The method generates bandwidth information, creates the physical topology of the network,
        and creates a dictionary containing all the necessary simulation parameters.

        :return: None
        """
        bw_info = create_bw_info(sim_type=self.properties['sim_type'])
        bw_file = f"bw_info_{self.properties['thread_num']}.json"

        self.save_input(file_name=bw_file, data=bw_info)

        path = f"./data/input/{self.properties['network']}/{self.properties['date']}/" \
               f"{self.properties['curr_time']}/{bw_file}"
        with open(path, 'r', encoding='utf-8') as file_object:
            self.properties['mod_per_bw'] = json.load(file_object)

        network_data = create_network(const_weight=self.properties['const_link_weight'],
                                      net_name=self.properties['network'])
        self.properties['topology_info'] = create_pt(cores_per_link=self.properties['cores_per_link'],
                                                     network_data=network_data)

    def run_yue(self):
        """
        Runs a simulation using the Yue's simulation assumptions. Reference: Wang, Yue. Dynamic Traffic Scheduling
        Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss. University of Massachusetts Lowell, 2022.

        :return: None
        """
        arr_rate_obj = self.properties['arrival_rate']
        start, stop, step = arr_rate_obj['start'], arr_rate_obj['stop'], arr_rate_obj['step']

        for arr_rate_mean in range(start, stop, step):
            arr_rate_mean = float(arr_rate_mean)

            self.properties['erlang'] = arr_rate_mean / self.properties['holding_time']
            arr_rate_mean *= float(self.properties['cores_per_link'])
            self.properties['arrival_rate'] = arr_rate_mean
            self.create_input()

            file_name = f"sim_input_{self.properties['thread_num']}.json"

            if arr_rate_mean == start:
                self.save_input(file_name=file_name, data=self.properties)

            self.properties['input_fp'] = f"./data/input/{self.properties['network']}/{self.properties['date']}/" \
                                          f"{self.properties['curr_time']}/{file_name}"
            engine = Engine(properties=self.properties)
            engine.run()

    def run_arash(self):
        """
        Runs a simulation using the Arash's simulation assumptions.
        Reference: https://doi.org/10.1016/j.comnet.2020.107755.

        :return: None
        """
        erlang_lst = [float(erlang) for erlang in range(50, 850, 50)]

        for erlang in erlang_lst:
            self.properties['arrival_rate'] = (self.properties['cores_per_link'] * erlang) / self.properties[
                'holding_time']

            self.properties['erlang'] = erlang

            self.create_input()

            if erlang == erlang_lst[0]:
                self.save_input(file_name=f"sim_input_{self.properties['thread_num']}.json", data=self.properties)

            engine = Engine(properties=self.properties)
            engine.run()

    def run_sim(self, **kwargs):
        """
        Controls the networking simulator class.

        :return: None
        """
        self.properties = kwargs['thread_params']
        # The date and current time derived from the simulation start
        self.properties['date'] = kwargs['sim_start'].split('_')[0]
        self.properties['curr_time'] = kwargs['sim_start'].split('_')[1]
        self.properties['sim_start'] = kwargs['sim_start']

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for thread_num, thread_params in threads_obj.items():
            sim_start = time.strftime("%m%d_%H:%M:%S")
            sim_obj = NetworkSimulator()
            class_inst = sim_obj.run_sim

            future = executor.submit(class_inst, thread_num=thread_num, thread_params=thread_params,
                                     sim_start=sim_start)

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == '__main__':
    threads_obj = read_config()
    run(threads_obj=threads_obj)
