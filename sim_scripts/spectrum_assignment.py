import itertools
from operator import itemgetter

import numpy as np

from arg_scripts.spectrum_args import empty_props
from helper_scripts.spectrum_helpers import SpectrumHelpers
from sim_scripts.snr_measurements import SnrMeasurements


class SpectrumAssignment:
    """
    Attempt to find the available spectrum for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: dict):
        self.spectrum_props = empty_props
        self.engine_props = engine_props
        self.sdn_props = sdn_props

        self.snr_obj = SnrMeasurements(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                       spectrum_props=self.spectrum_props)
        self.spec_help_obj = SpectrumHelpers(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                             spectrum_props=self.spectrum_props)

    def _allocate_best_fit(self, channels_list: list):
        for channel_dict in channels_list:
            for start_index in channel_dict['channel']:
                end_index = (start_index + self.spectrum_props['slots_needed'] + self.engine_props['guard_slots']) - 1
                if end_index not in channel_dict['channel']:
                    break

                if len(self.spectrum_props['path_list']) > 2:
                    self.spec_help_obj.start_index = start_index
                    self.spec_help_obj.end_index = end_index
                    self.spec_help_obj.core_num = channel_dict['core']
                    self.spec_help_obj.check_other_links()

                if self.spectrum_props['is_free'] or len(self.spectrum_props['path_list']) <= 2:
                    self.spectrum_props['is_free'] = True
                    self.spectrum_props['start_slot'] = start_index
                    self.spectrum_props['end_slot'] = end_index + self.engine_props['guard_slots']
                    self.spectrum_props['core_num'] = channel_dict['core']
                    return

    def find_best_fit(self):
        """
        Searches for and allocates the best-fit super channel on each link along the path.
        """
        channels_list = list()

        # Get all potential super channels
        for (src, dest) in zip(self.spectrum_props['path_list'][:-1], self.spectrum_props['path_list'][1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                if self.spectrum_props['forced_core'] is not None and self.spectrum_props['forced_core'] != core_num:
                    continue

                core_arr = self.sdn_props['net_spec_dict'][(src, dest)]['cores_matrix'][core_num]
                open_slots_arr = np.where(core_arr == 0)[0]

                tmp_matrix = [list(map(itemgetter(1), g)) for k, g in
                              itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                for channel_list in tmp_matrix:
                    if len(channel_list) >= self.spectrum_props['slots_needed']:
                        channels_list.append({'link': (src, dest), 'core': core_num, 'channel': channel_list})

        # Sort the list of candidate super channels
        channels_list = sorted(channels_list, key=lambda d: len(d['channel']))
        self._allocate_best_fit(channels_list=channels_list)

    def _setup_first_last(self):
        if self.spectrum_props['forced_core'] is not None:
            core_matrix = [self.spectrum_props['cores_matrix'][self.spectrum_props['forced_core']]]
            core_list = [self.spectrum_props['forced_core']]
        elif self.engine_props['allocation_method'] in ('priority_first', 'priority_last'):
            core_list = [0, 2, 4, 1, 3, 5, 6]
            core_matrix = list()
            for curr_core in core_list:
                core_matrix.append(self.spectrum_props['cores_matrix'][curr_core])
        else:
            core_matrix = self.spectrum_props['cores_matrix']
            core_list = list(range(0, self.engine_props['cores_per_link']))

        return core_matrix, core_list

    def handle_first_last(self, flag: str):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method to be used.
        """
        core_matrix, core_list = self._setup_first_last()
        for core_arr, core_num in zip(core_matrix, core_list):
            open_slots_arr = np.where(core_arr == 0)[0]

            # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
            if flag in ('last_fit', 'priority_last'):
                open_slots_matrix = [list(map(itemgetter(1), g))[::-1] for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
            elif flag in ('first_fit', 'priority_first', 'forced_index'):
                open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
            else:
                raise NotImplementedError(f'Invalid flag, got: {flag} and expected last_fit or first_fit.')

            self.spec_help_obj.core_num = core_num
            was_allocated = self.spec_help_obj.check_super_channels(open_slots_matrix=open_slots_matrix, flag=flag)
            if was_allocated:
                return

    # fixme: Only works for 7 cores
    def xt_aware(self):
        """
        Attempts to allocate a request with the least amount of cross-talk interference on neighboring cores.

        :return: The information of the request if allocated or False if not possible.
        :rtype: dict
        """
        core = self.spec_help_obj.find_best_core()
        # Graph coloring for cores will be in this order
        if core in [0, 2, 4, 6]:
            self.spectrum_props['forced_core'] = core
            return self.handle_first_last(flag='first_fit')

        return self.handle_first_last(flag='last_fit')

    def _get_spectrum(self, ai_obj: object):
        if self.spectrum_props['forced_index'] is not None:
            self.handle_first_last(flag='forced_index')
        elif self.engine_props['allocation_method'] == 'best_fit':
            self.find_best_fit()
        elif self.engine_props['allocation_method'] in ('first_fit', 'last_fit', 'priority_first', 'priority_last'):
            self.handle_first_last(flag=self.engine_props['allocation_method'])
        elif self.engine_props['allocation_method'] == 'xt_aware':
            self.xt_aware()
        else:
            raise NotImplementedError(f"Expected first_fit or best_fit, got: {self.engine_props['allocation_method']}")

    def _init_spectrum_info(self):
        link_tuple = (self.spectrum_props['path_list'][0], self.spectrum_props['path_list'][1])
        rev_link_tuple = (self.spectrum_props['path_list'][1], self.spectrum_props['path_list'][0])
        self.spectrum_props['cores_matrix'] = self.sdn_props['net_spec_dict'][link_tuple]['cores_matrix']
        self.spectrum_props['rev_cores_matrix'] = self.sdn_props['net_spec_dict'][rev_link_tuple]['cores_matrix']
        self.spectrum_props['is_free'] = False

    def get_spectrum(self, mod_format_list: list, ai_obj: object, slice_bandwidth: str = None):
        """
        Controls the class, attempts to find an available spectrum.

        :param mod_format_list: A list of modulation formats to attempt allocation.
        :param ai_obj: An artificial intelligence class.
        :param slice_bandwidth: A bandwidth used for light-segment slicing.
        """
        self._init_spectrum_info()
        for modulation in mod_format_list:
            if modulation is False:
                self.sdn_props['block_reason'] = 'distance'
                continue

            if slice_bandwidth:
                bandwidth_dict = self.engine_props['mod_per_bw'][slice_bandwidth]
                self.spectrum_props['slots_needed'] = bandwidth_dict[modulation]['slots_needed']
            else:
                self.spectrum_props['slots_needed'] = self.sdn_props['mod_formats'][modulation]['slots_needed']
            self._get_spectrum(ai_obj=ai_obj)

            if self.spectrum_props['is_free']:
                self.spectrum_props['modulation'] = modulation
                if self.engine_props['snr_type'] != 'None' and self.engine_props['snr_type'] is not None:
                    snr_check, xt_cost = self.snr_obj.handle_snr()
                    self.spectrum_props['xt_cost'] = xt_cost
                    if not snr_check:
                        self.spectrum_props['is_free'] = False
                        self.sdn_props['block_reason'] = 'xt_threshold'
                        continue

                    self.spectrum_props['is_free'] = True
                    self.sdn_props['block_reason'] = None

                return

            self.spectrum_props['block_reason'] = 'congestion'
            continue
