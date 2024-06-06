# TODO: This might break into its own module
import os
import json

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

sims_info_dict = find_times(dates_dict={'0606': 'NSFNet'}, filter_dict=filter_dict)
helpers_obj = PlotHelpers(plot_props=empty_props, net_names_list=sims_info_dict['networks_matrix'])
helpers_obj.get_file_info(sims_info_dict=sims_info_dict)


def read_files():
    output_fp = os.path.join('..', 'data', 'output', network, date, run_time, 's1',
                             f'{erlang}_erlang.json')
    input_fp = os.path.join('..', 'data', 'input', network, date, run_time, f'sim_input_s1.json')
    with open(output_fp, 'r') as file_path:
        output_dict = json.load(file_path)
    with open(input_fp, 'r') as file_path:
        input_dict = json.load(file_path)

    return input_dict, output_dict


# TODO: We will only have 's1' for now
for run_time, run_obj in helpers_obj.file_info.items():
    net_key, date_key, sim_key = list(run_obj.keys())
    network = run_obj[net_key]
    date = run_obj[date_key]
    sim_dict = run_obj[sim_key]
    erlang = sim_dict['s1'][0]

    input_dict, output_dict = read_files()
    print('Here')
