import json
import os
import math
import copy
from statistics import mean, variance, stdev

import numpy as np
import pandas as pd

from arg_scripts.stats_args import empty_props
from arg_scripts.stats_args import SNAP_KEYS_LIST
from helper_scripts.sim_helpers import find_path_len, find_core_cong, get_hfrag
from helper_scripts.os_helpers import create_dir

class SupervisedStats:
    """
    The SupervisedStats class finds and stores all relevant statistics for trainign data sets of supervised learning in simulations.
    """

    def __init__(self, engine_props: dict, sim_info: str, saving_data_type: str ):

        self.engine_props = engine_props
        self.sim_info = sim_info

        self.save_dict = {'iter_stats': {}}

        # Used to check the training data type
        self.saving_data_type = saving_data_type

        # TODO: Make sure this isn't reset after multiple iterations
        self.train_data_list = list()

    
    def update_train_slicing_data(self, old_req_info_dict: dict, req_info_dict: dict, net_spec_dict: dict,
                                  curr_trans: int, save: bool, base_fp: str):
        """
        Updates the training data list with the current request information.

        :param old_req_info_dict: Request dictionary before any potential slicing.
        :param req_info_dict: Request dictionary after potential slicing.
        :param net_spec_dict: Network spectrum database.
        :param save: Flag to save data on file.
        :param base_fp: Bade path address for saving file
        """
        path_list = req_info_dict['path']
        cong_arr = np.array([])

        for core_num in range(self.engine_props['cores_per_link']):
            curr_cong = find_core_cong(core_index=core_num, net_spec_dict=net_spec_dict, path_list=path_list)
            cong_arr = np.append(cong_arr, curr_cong)

        path_length = find_path_len(path_list=path_list, topology=self.engine_props['topology'])
        tmp_info_dict = {
            'old_bandwidth': old_req_info_dict['bandwidth'],
            'path_length': path_length,
            'longest_reach': np.max(old_req_info_dict['mod_formats']['QPSK']['max_length']),
            'ave_cong': float(np.mean(cong_arr)),
            'num_segments': curr_trans,
        }
        self.train_data_list.append(tmp_info_dict)
        if save:
            save_df = pd.DataFrame(self.train_data_list)
            save_df.to_csv(f"{base_fp}/output/{self.sim_info}/{self.engine_props['erlang']}_train_data.csv",
                           index=False)
    

    def update_train_xtar_data(self, train_list: list, save: bool, base_fp: str):
        """
        Updates the training data list with the current request information.

        :param train_list: List of link statuses and xt cost.
        :param save: Flag to save data on file.
        :param base_fp: Bade path address for saving file
        """
        self.train_data_list.extend(train_list)
        if save == True:
            save_df = pd.DataFrame(self.train_data_list)
            file_path = f"{base_fp}/output/{self.sim_info}/{self.engine_props['erlang']}_train_xtar_data.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            self.train_data_list.clear()
            if not os.path.isfile(file_path):
                save_df.to_csv(file_path, index=False)
            else:
                save_df.to_csv(file_path, mode='a', header=False, index=False)
            
            self.train_data_list.clear()

