# pylint: disable=cell-var-from-loop

import os
import json

import numpy as np
import pandas as pd

from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import find_times, PlotHelpers
from arg_scripts.plot_args import PlotProps

NETWORK_LIST = ['Pan-European']
ARRIVAL_RATE_LIST = [80, 100, 140, 180]

inf_baselines = {
    ('NSFNet', 80): 0.0,
    ('NSFNet', 100): 0.0005,
    ('NSFNet', 140): 0.0335,
    ('NSFNet', 180): 0.071,

    ('USNet', 80): 0.011,
    ('USNet', 100): 0.0115,
    ('USNet', 140): 0.0175,
    ('USNet', 180): 0.024,

    # 4,000 requests and 500 episodes
    ('Pan-European', 80): 0.0,
    ('Pan-European', 100): 0.0,
    ('Pan-European', 140): 0.0187,
    ('Pan-European', 180): 0.0512,
}

ksp_baselines = {
    ('NSFNet', 80): 0.0215,
    ('NSFNet', 100): 0.0505,
    ('NSFNet', 140): 0.0735,
    ('NSFNet', 180): 0.1065,

    ('USNet', 80): 0.0295,
    ('USNet', 100): 0.035,
    ('USNet', 140): 0.051,
    ('USNet', 180): 0.0715,

    # 4,000 requests and 500 episodes
    ('Pan-European', 80): 0.0013,
    ('Pan-European', 100): 0.006,
    ('Pan-European', 140): 0.0387,
    ('Pan-European', 180): 0.0767,
}

spf_baselines = {
    ('NSFNet', 80): 0.081,
    ('NSFNet', 100): 0.128,
    ('NSFNet', 140): 0.168,
    ('NSFNet', 180): 0.208,

    ('USNet', 80): 0.056,
    ('USNet', 100): 0.071,
    ('USNet', 140): 0.0965,
    ('USNet', 180): 0.111,

    # 4,000 requests and 500 episodes
    ('Pan-European', 80): 0.005,
    ('Pan-European', 100): 0.0227,
    ('Pan-European', 140): 0.068,
    ('Pan-European', 180): 0.1062,
}

for network in NETWORK_LIST:
    for arrival_rate in ARRIVAL_RATE_LIST:
        filter_dict = {
            'and_filter_list': [
                ['arrival_rate', 'start', arrival_rate],
                # TODO: Now filter function doesn't work
                # ['max_iters', 400],
            ],
            'or_filter_list': [
            ],
            'not_filter_list': [
                # ['max_segments', 4],
                # ['max_segments', 8],
            ]
        }

        sims_info_dict = find_times(dates_dict={'1002': network},
                                    filter_dict=filter_dict)
        helpers_obj = PlotHelpers(plot_props=PlotProps(), net_names_list=sims_info_dict['networks_matrix'])
        helpers_obj.get_file_info(sims_info_dict=sims_info_dict)

        counter = 0  # pylint: disable=invalid-name
        dict_list = []
        BATCH_SIZE = 200

        save_fp = os.path.join('..', 'data', 'excel')
        create_dir(file_path=save_fp)
        csv_file = os.path.join(save_fp, f'{network}_analysis_{arrival_rate / 0.2}.csv')


        def read_files():
            """
            Reads a file from a single reinforcement learning simulation run.

            :return: The input and output JSON files.
            :rtype: tuple
            """
            output_fp = os.path.join('..', 'data', 'output', network, date, run_time, 's1',
                                     f'{erlang}_erlang.json')
            input_fp = os.path.join('..', 'data', 'input', network, date, run_time, 'sim_input_s1.json')

            if not os.path.exists(output_fp):
                print(f'Output file not found! Skipping: {output_fp}')
                return False, False
            if not os.path.exists(input_fp):
                print(f'Input file not found! Skipping: {input_fp}')
                return False, False

            with open(output_fp, 'r', encoding='utf-8') as file_path:
                try:
                    output_dict = json.load(file_path)
                except json.JSONDecodeError:
                    print('Output dict not found.')
                    return False, False
            with open(input_fp, 'r', encoding='utf-8') as file_path:
                try:
                    input_dict = json.load(file_path)
                except json.JSONDecodeError:
                    print('Input dict not found.')
                    return False, False

            return input_dict, output_dict


        def calculate_baseline_reductions(tmp_dict, network, arrival_rate):
            """
            Calculates percentage reductions compared to SPF, KSP, and KSP-Inf baselines.
            Updates tmp_dict in-place.

            :param tmp_dict: Dictionary containing 'Blocking' key.
            :param network: The network name.
            :param arrival_rate: The arrival rate.
            """
            # Calculate percentage reductions compared to SPF baseline
            spf_baseline = spf_baselines.get((network, arrival_rate))
            if spf_baseline is not None and spf_baseline != 0:
                spf_reduction = ((spf_baseline - tmp_dict['Blocking']) / spf_baseline) * 100
                tmp_dict['SPF Reduction (%)'] = spf_reduction
            else:
                tmp_dict['SPF Reduction (%)'] = np.inf

            # Calculate percentage reductions compared to KSP baseline
            ksp_baseline = ksp_baselines.get((network, arrival_rate))
            if ksp_baseline is not None and ksp_baseline != 0:
                ksp_reduction = ((ksp_baseline - tmp_dict['Blocking']) / ksp_baseline) * 100
                tmp_dict['KSP Reduction (%)'] = ksp_reduction
            else:
                tmp_dict['KSP Reduction (%)'] = np.inf

            # Calculate percentage reductions compared to KSP-Inf baseline
            inf_baseline = inf_baselines.get((network, arrival_rate))
            if inf_baseline is not None and inf_baseline != 0:
                inf_reduction = ((inf_baseline - tmp_dict['Blocking']) / inf_baseline) * 100
                tmp_dict['KSP-Inf Reduction (%)'] = inf_reduction
            else:
                tmp_dict['KSP-Inf Reduction (%)'] = np.inf


        def get_dict(input_dict: dict, output_dict: dict, network: str, arrival_rate: int):
            """
            Gets desired information from input and output for a single RL simulation.
            Calculates percentage reductions compared to SPF and KSP baselines.

            :param input_dict: Input dictionary.
            :param output_dict: Output dictionary.
            :param network: The network name.
            :param arrival_rate: The arrival rate.
            :return: Relevant data from input/output with percentage reductions.
            :rtype: dict
            """
            tmp_dict = dict()
            last_key = list(output_dict['iter_stats'].keys())[-1]
            tmp_dict['Blocking'] = np.mean(output_dict['iter_stats'][last_key]['sim_block_list'][-10:])
            tmp_dict['Completed Iters'] = len(output_dict['iter_stats'][last_key]['sim_block_list'])
            tmp_dict['Sim Start'] = input_dict['sim_start'].split('_')[-1]

            tmp_dict['Alpha Start'] = input_dict['alpha_start']
            tmp_dict['Alpha End'] = input_dict['alpha_end']
            tmp_dict['Alpha Update'] = input_dict['alpha_update']

            tmp_dict['Epsilon Start'] = input_dict['epsilon_start']
            tmp_dict['Epsilon End'] = input_dict['epsilon_end']
            tmp_dict['Epsilon Update'] = input_dict['epsilon_update']

            tmp_dict['Reward'] = input_dict['reward']
            tmp_dict['Penalty'] = input_dict['penalty']
            tmp_dict['Path Algorithm'] = input_dict['path_algorithm']

            calculate_baseline_reductions(tmp_dict, network, arrival_rate)

            return tmp_dict


        if not os.path.exists(csv_file):  # pylint: disable=simplifiable-if-statement
            HEADER = True
        else:
            HEADER = False

        # TODO: Only supports 's1'
        for run_time, run_obj in helpers_obj.file_info.items():
            net_key, date_key, sim_key = list(run_obj.keys())
            sim_network = run_obj[net_key]
            date = run_obj[date_key]
            sim_dict = run_obj[sim_key]

            try:
                erlang = sim_dict['s1'][0]
            except KeyError:
                print(f'No data found in dictionary. Skipping: {run_time}')
                continue

            input_dict, output_dict = read_files()

            if not input_dict or not output_dict:
                continue
            if '19' in input_dict['sim_start'][0:2]:
                continue

            tmp_dict = get_dict(
                input_dict=input_dict,
                output_dict=output_dict,
                network=network,
                arrival_rate=arrival_rate
            )
            dict_list.append(tmp_dict)
            counter += 1

            print(f'{counter} dictionaries created in this batch.')
            if counter == BATCH_SIZE:
                print(f'Completed one batch of {BATCH_SIZE}, appending to a CSV!')
                df_to_write = pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list])
                df_to_write.to_csv(
                    csv_file, mode='a', index=False, header=HEADER, encoding='utf-8')
                counter = 0  # pylint: disable=invalid-name
                dict_list = []
                HEADER = False

        if dict_list:
            pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(
                csv_file, mode='a', index=False, header=HEADER, encoding='utf-8')

        df = pd.read_csv(csv_file, encoding='utf-8')
        df = df.sort_values(by='Blocking')
        df.to_csv(csv_file, index=False, encoding='utf-8')
