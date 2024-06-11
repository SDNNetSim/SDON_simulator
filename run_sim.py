# Standard library imports
import time
import copy
import os
from datetime import datetime

# Third-party library imports
import concurrent.futures

# Local application imports
from helper_scripts.setup_helpers import create_input, save_input
from sim_scripts.engine import Engine
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args


class NetworkSimulator:
    """
    Controls all simulations for this project.
    """

    def __init__(self):
        """
        Initializes the NetworkSimulator class.
        """
        # Contains all the desired network simulator parameters for every simulation
        self.properties = None

    def _run_yue(self, arr_rate_mean: int, start: int):
        arr_rate_mean = float(arr_rate_mean)
        engine_props = copy.deepcopy(self.properties)
        engine_props['erlang'] = arr_rate_mean / engine_props['holding_time']
        arr_rate_mean *= float(engine_props['cores_per_link'])
        engine_props['arrival_rate'] = arr_rate_mean
        create_input(engine_props=engine_props, base_fp='data')

        if arr_rate_mean == (start * engine_props['cores_per_link']):
            file_name = f"sim_input_{self.properties['thread_num']}.json"
            save_input(base_fp='data', properties=engine_props, file_name=file_name, data_dict=engine_props)

        engine = Engine(engine_props=engine_props)
        engine.run()

    def run_yue(self):
        """
        Runs a simulation using the Yue's simulation assumptions. Reference: Wang, Yue. Dynamic Traffic Scheduling
        Frameworks with Spectral and Spatial Flexibility in Sdm-Eons. Diss. University of Massachusetts Lowell, 2022.
        """
        arr_rate_dict = self.properties['arrival_rate']
        start, stop, step = arr_rate_dict['start'], arr_rate_dict['stop'], arr_rate_dict['step']

        if self.properties['thread_erlangs']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures_list = []
                for arr_rate_mean in range(start, stop, step):
                    time.sleep(1.0)
                    future = executor.submit(self._run_yue, arr_rate_mean=arr_rate_mean, start=start)
                    futures_list.append(future)

                for future in concurrent.futures.as_completed(futures_list):
                    future.result()
        else:
            for arr_rate_mean in range(start, stop, step):
                self._run_yue(arr_rate_mean=arr_rate_mean, start=start)

    def _run_arash(self, erlang: float, first_erlang: float):
        engine_props = copy.deepcopy(self.properties)
        engine_props['arrival_rate'] = (engine_props['cores_per_link'] * erlang) / engine_props['holding_time']
        engine_props['erlang'] = erlang
        local_props = create_input(base_fp='data', engine_props=engine_props)

        if first_erlang:
            save_input(base_fp='data', properties=engine_props, file_name=f"sim_input_{local_props['thread_num']}.json",
                       data_dict=local_props)

        engine = Engine(engine_props=engine_props)
        engine.run()

    def run_arash(self):
        """
        Runs a simulation using the Arash's simulation assumptions.
        Reference: https://doi.org/10.1016/j.comnet.2020.107755.
        """
        erlang_dict = self.properties['erlangs']
        start, stop, step = erlang_dict['start'], erlang_dict['stop'], erlang_dict['step']
        erlang_list = [float(erlang) for erlang in range(start, stop, step)]

        if self.properties['thread_erlangs']:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures_list = []
                for erlang in erlang_list:
                    first_erlang = erlang == erlang_list[0]
                    time.sleep(1.0)
                    future = executor.submit(self._run_arash, erlang=erlang, first_erlang=first_erlang)
                    futures_list.append(future)

                for future in concurrent.futures.as_completed(futures_list):
                    future.result()
        else:
            for erlang in erlang_list:
                first_erlang = erlang == erlang_list[0]
                self._run_arash(erlang=erlang, first_erlang=first_erlang)

    def run_sim(self, **kwargs):
        """
        Runs all simulations.
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
        else:
            self.run_arash()


def run(sims_dict: dict):
    """
    Runs multiple simulations concurrently or a single simulation.

    :param sims_dict: Contains the parameters for each simulation.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_list = []

        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        for thread_num, thread_params in sims_dict.items():
            curr_sim = NetworkSimulator()
            class_inst = curr_sim.run_sim

            time.sleep(1.0)
            future = executor.submit(class_inst, thread_num=thread_num, thread_params=thread_params,
                                     sim_start=sim_start)

            futures_list.append(future)

        for future in concurrent.futures.as_completed(futures_list):
            future.result()


if __name__ == '__main__':
    args_obj = parse_args()
    all_sims_dict = read_config(args_obj=args_obj)
    run(sims_dict=all_sims_dict)
