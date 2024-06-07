# TODO: This might break into its own module
import os
import json

import numpy as np
import pandas as pd

from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import find_times, PlotHelpers
from arg_scripts.plot_args import empty_props

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

sims_info_dict = find_times(dates_dict={'0606': 'NSFNet', '0607': 'NSFNet'}, filter_dict=filter_dict)
helpers_obj = PlotHelpers(plot_props=empty_props, net_names_list=sims_info_dict['networks_matrix'])
helpers_obj.get_file_info(sims_info_dict=sims_info_dict)

counter = 0
dict_list = []
batch_size = 200

save_fp = os.path.join('..', 'data', 'output', 'excel')
create_dir(file_path=save_fp)
csv_file = os.path.join(save_fp, 'analysis.csv')


def read_files():
    output_fp = os.path.join('..', 'data', 'output', network, date, run_time, 's1',
                             f'{erlang}_erlang.json')
    input_fp = os.path.join('..', 'data', 'input', network, date, run_time, f'sim_input_s1.json')
    try:
        with open(output_fp, 'r') as file_path:
            output_dict = json.load(file_path)
    except FileNotFoundError:
        print(f'Input file found but not an output file! Skipping: {output_fp}')
        return False, False
    with open(input_fp, 'r') as file_path:
        input_dict = json.load(file_path)

    return input_dict, output_dict


def get_dict(input_dict: dict, output_dict: dict):
    tmp_dict = dict()
    last_key = list(output_dict['iter_stats'].keys())[-1]
    tmp_dict['Blocking'] = np.mean(output_dict['iter_stats'][last_key]['sim_block_list'][-5:])
    tmp_dict['Completed Iters'] = len(output_dict['iter_stats'][last_key]['sim_block_list'])

    # tmp_dict['Mean Hops'] = output_dict['hops_mean']
    # tmp_dict['Min Hops'] = output_dict['hops_min']
    # tmp_dict['Max Hops'] = output_dict['hops_max']
    #
    # tmp_dict['Mean Length'] = output_dict['lengths_mean']
    # tmp_dict['Min Length'] = output_dict['lengths_min']
    # tmp_dict['Max Length'] = output_dict['lengths_max']

    # Parameters
    tmp_dict['Learning Rate'] = input_dict['learn_rate']
    tmp_dict['Discount Factor'] = input_dict['discount_factor']
    tmp_dict['Reward'] = input_dict['reward']
    tmp_dict['Penalty'] = input_dict['penalty']
    tmp_dict['Epsilon Start'] = input_dict['epsilon_start']
    tmp_dict['Epsilon End'] = input_dict['epsilon_end']

    return tmp_dict


# TODO: We will only have 's1' for now
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
    if not input_dict:
        continue

    tmp_dict = get_dict(input_dict=input_dict, output_dict=output_dict)
    dict_list.append(tmp_dict)
    counter += 1

    print(f'{counter} dictionaries created in this batch.')
    if counter == batch_size:
        print(f'Completed one batch of {batch_size}, appending to a CSV!')
        pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(csv_file, mode='a', index=False)
        counter = 0
        dict_list = []

if dict_list:
    pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(csv_file, mode='a', index=False)

df = pd.read_csv(csv_file)
df = df.sort_values(by='Blocking')
df.to_csv(csv_file, index=False)
