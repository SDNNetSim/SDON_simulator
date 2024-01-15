import itertools
from operator import itemgetter

import numpy as np

from arg_scripts.spectrum_args import empty_props
from helper_scripts.spectrum_helpers import check_open_slots
from helper_scripts.sim_helpers import handle_snr
from sim_scripts.snr_measurements import SnrMeasurements


# TODO: Move a lot of these to spectrum helpers/args
# TODO: Naming conventions for variables and functions
# TODO: Instead of breaking up, move stuff to another file if you can
# TODO: If pylint too few public methods, then rename some methods
# TODO: AI object should be called here for a spectrum
# TODO: Make sure to reset these properly for each request
# TODO: Naming conventions!
# TODO: What should go in a dict and what shouldn't?
class SpectrumAssignment:
    """
    Finds the available spectrum for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: dict, route_props: dict):
        self.spectrum_props = empty_props
        self.engine_props = engine_props
        self.sdn_props = sdn_props

        self.snr_obj = SnrMeasurements(properties=self.sdn_props)

    # TODO: Break this up into at least 2 functions
    # TODO: This does not work
    # TODO: Check to see if this works after you finish spectrum assignment updates
    def _best_fit_allocation(self):
        """
        Searches for and allocates the best-fit super channel on each link along the path.
        """
        res_list = []

        # Get all potential super channels
        for (src, dest) in zip(self.spectrum_props['path_list'][:-1], self.spectrum_props['path_list'][1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.sdn_props['net_spec_dict'][(src, dest)]['cores_matrix'][core_num]
                open_slots_arr = np.where(core_arr == 0)[0]

                tmp_matrix = [list(map(itemgetter(1), g)) for k, g in
                              itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                for channel in tmp_matrix:
                    if len(channel) >= self.spectrum_props['slots_needed']:
                        res_list.append({'link': (src, dest), 'core': core_num, 'channel': channel})

        # Sort the list of candidate super channels
        sorted_list = sorted(res_list, key=lambda d: len(d['channel']))

        for channel_dict in sorted_list:
            for start_index in channel_dict['channel']:
                end_index = (start_index + self.spectrum_props['slots_needed'] + self.engine_props['guard_slots']) - 1
                if end_index not in channel_dict['channel']:
                    break

                if len(self.spectrum_props['path_list']) > 2:
                    self._check_other_links(channel_dict['core'], start_index,
                                            end_index + self.engine_props['guard_slots'])

                if self.is_free is not False or len(self.spectrum_props['path_list']) <= 2:
                    self.response = {'core_num': channel_dict['core'], 'start_slot': start_index,
                                     'end_slot': end_index + self.engine_props['guard_slots']}
                    return

    def _handle_first_last(self, flag: str):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method to be used.
        :type flag: str
        """
        if self.spectrum_props['forced_core'] is not None:
            matrix = [self.spectrum_props['cores_matrix'][self.spectrum_props['forced_core']]]
            start = self.spectrum_props['forced_core']
        else:
            matrix = self.spectrum_props['cores_matrix']
            start = 0

        for core_num, core_arr in enumerate(matrix, start=start):
            open_slots_arr = np.where(core_arr == 0)[0]

            # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
            if flag == 'last_fit':
                open_slots_matrix = [list(map(itemgetter(1), g))[::-1] for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
            elif flag == 'first_fit':
                open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
            else:
                raise NotImplementedError(f'Invalid flag, got: {flag} and expected last_fit or first_fit')

            was_allocated = check_open_slots(sdn_props=self.sdn_props, spectrum_props=self.spectrum_props,
                                             engine_props=self.engine_props,
                                             open_slots_matrix=open_slots_matrix, core_num=core_num)
            if was_allocated:
                return

    # fixme: Only works for 7 cores
    def _xt_aware_allocation(self):
        """
        Attempts to allocate a request with the least amount of cross-talk interference on neighboring cores.

        :return: The information of the request if allocated or False if not possible.
        :rtype dict
        """
        core = self.spec_help_obj.find_best_core()
        # Graph coloring for cores will be in this order
        if core in [0, 2, 4, 6]:
            self.spectrum_props['forced_core'] = core
            return self._handle_first_last(flag='first_fit')

        return self._handle_first_last(flag='last_fit')

    def _get_spectrum(self):
        if self.engine_props['allocation_method'] == 'best_fit':
            self._best_fit_allocation()
        elif self.engine_props['allocation_method'] in ('first_fit', 'last_fit'):
            self._handle_first_last(flag=self.engine_props['allocation_method'])
        elif self.engine_props['allocation_method'] == 'xt_aware':
            self._xt_aware_allocation()
        else:
            raise NotImplementedError(f"Expected first_fit or best_fit, got: {self.engine_props['allocation_method']}")

    def _init_spectrum_info(self):
        link_tuple = (self.spectrum_props['path_list'][0], self.spectrum_props['path_list'][1])
        rev_link_tuple = (self.spectrum_props['path_list'][1], self.spectrum_props['path_list'][0])
        self.spectrum_props['cores_matrix'] = self.sdn_props['net_spec_dict'][link_tuple]['cores_matrix']
        self.spectrum_props['rev_cores_matrix'] = self.sdn_props['net_spec_dict'][rev_link_tuple]['cores_matrix']
        self.spectrum_props['is_free'] = False

    def get_spectrum(self, mod_options: list):
        """
        Controls the class, attempts to find an available spectrum.

        :param mod_options: A list of modulation formats to attempt allocation.
        """
        self._init_spectrum_info()
        for modulation in mod_options:
            if modulation is False:
                # TODO: Light segment slicing will be a helper function for spectrum assignment
                if self.engine_props['max_segments'] > 1:
                    raise NotImplementedError

                continue

            self.spectrum_props['slots_needed'] = self.sdn_props['mod_formats'][modulation]['slots_needed']
            self._get_spectrum()
            if self.spectrum_props['is_free'] is not False:
                self.spectrum_props['modulation'] = modulation
                if self.engine_props['check_snr'] != 'None' and self.engine_props['check_snr'] is not None:
                    # TODO: Fix this
                    snr_check, xt_cost = handle_snr(engine_props=self.engine_props, sdn_props=self.sdn_props)
                    # TODO: Make sure to account for this in sdn_controller
                    self.spectrum_props['xt_cost'] = xt_cost
                    if not snr_check:
                        self.spectrum_props['is_free'] = False
                        self.sdn_props['block_reason'] = 'xt_threshold'
                        return

                return

            self.spectrum_props['block_reason'] = 'congestion'
            return
