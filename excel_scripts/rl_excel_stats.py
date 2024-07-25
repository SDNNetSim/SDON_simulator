import os
import json

import numpy as np
import pandas as pd

from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import find_times, PlotHelpers
from arg_scripts.plot_args import PlotProps

filter_dict = {
    'and_filter_list': [
    ],
    'or_filter_list': [
    ],
    'not_filter_list': [
        # ['max_segments', 4],
        # ['max_segments', 8],
    ]
}

sims_info_dict = find_times(dates_dict={'0613': 'NSFNet'}, filter_dict=filter_dict)
helpers_obj = PlotHelpers(plot_props=PlotProps(), net_names_list=sims_info_dict['networks_matrix'])
helpers_obj.get_file_info(sims_info_dict=sims_info_dict)

counter = 0  # pylint: disable=invalid-name
dict_list = []
BATCH_SIZE = 200

save_fp = os.path.join('..', 'data', 'excel')
create_dir(file_path=save_fp)
csv_file = os.path.join(save_fp, 'analysis.csv')


def read_files():
    """
    Reads a file from a single reinforcement learning simulation run.

    :return: The input and output JSON files.
    :rtype: tuple
    """
    output_fp = os.path.join('..', 'data', 'output', network, date, run_time, 's1',
                             f'{erlang}_erlang.json')
    input_fp = os.path.join('..', 'data', 'input', network, date, run_time, 'sim_input_s1.json')
    try:
        with open(output_fp, 'r', encoding='utf-8') as file_path:
            output_dict = json.load(file_path)
    except FileNotFoundError:
        print(f'Input file found but not an output file! Skipping: {output_fp}')
        return False, False
    with open(input_fp, 'r', encoding='utf-8') as file_path:
        input_dict = json.load(file_path)

    return input_dict, output_dict


def get_dict(input_dict: dict, output_dict: dict):
    """
    Gets desired information from input and output for a single RL simulation.

    :param input_dict: Input dictionary.
    :param output_dict: Output dictionary.
    :return: Relevant data from input/output.
    :rtype: dict
    """
    tmp_dict = dict()
    last_key = list(output_dict['iter_stats'].keys())[-1]
    tmp_dict['Blocking'] = np.mean(output_dict['iter_stats'][last_key]['sim_block_list'][-10:])
    tmp_dict['Completed Iters'] = len(output_dict['iter_stats'][last_key]['sim_block_list'])
    tmp_dict['Sim Start'] = input_dict['sim_start'].split('_')[-1]

    tmp_dict['Learning Rate'] = input_dict['learn_rate']
    tmp_dict['Discount Factor'] = input_dict['discount_factor']
    tmp_dict['Reward'] = input_dict['reward']
    tmp_dict['Penalty'] = input_dict['penalty']
    tmp_dict['Epsilon Start'] = input_dict['epsilon_start']
    tmp_dict['Epsilon End'] = input_dict['epsilon_end']
    tmp_dict['Path Algorithm'] = input_dict['path_algorithm']
    tmp_dict['Core Algorithm'] = input_dict['core_algorithm']

    return tmp_dict


# TODO: Only supports 's1'
for run_time, run_obj in helpers_obj.file_info.items():
    net_key, date_key, sim_key = list(run_obj.keys())
    network = run_obj[net_key]
    date = run_obj[date_key]
    sim_dict = run_obj[sim_key]

    try:
        erlang = sim_dict['s1'][0]
    except KeyError:
        print(f'No data found in dictionary. Skipping: {run_time}')

    input_dict, output_dict = read_files()

    if '19' in input_dict['sim_start'][0:2]:
        continue
    if not input_dict:
        continue

    tmp_dict = get_dict(input_dict=input_dict, output_dict=output_dict)
    dict_list.append(tmp_dict)
    counter += 1

    print(f'{counter} dictionaries created in this batch.')
    if counter == BATCH_SIZE:
        print(f'Completed one batch of {BATCH_SIZE}, appending to a CSV!')
        pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(csv_file, mode='a', index=False)
        counter = 0  # pylint: disable=invalid-name
        dict_list = []

if dict_list:
    pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(csv_file, mode='a', index=False)

df = pd.read_csv(csv_file)
df = df.sort_values(by='Blocking')
df.to_csv(csv_file, index=False)
