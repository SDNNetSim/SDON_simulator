import itertools
import warnings
from operator import itemgetter

import numpy as np

from arg_scripts.spectrum_args import empty_props
from helper_scripts.sim_helpers import update_snr_obj, handle_snr
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
        # TODO: Probably won't need route props
        self.route_props = route_props

        # TODO: How will this be updated?
        self.snr_obj = SnrMeasurements(properties=self.sdn_props)

    # TODO: Make sure to check link and rev_link
    # TODO: Can put multiple params in constructor as well
    def _link_has_free_spectrum(self, link, rev_link, core_num, start_slot, end_slot):
        spec = self.sdn_props['net_spec_dict'][link]['cores_matrix'][core_num][start_slot:end_slot]
        rev_spec = self.sdn_props['net_spec_dict'][rev_link]['cores_matrix'][core_num][start_slot:end_slot]

        if set(spec) == {0.0} and set(rev_spec) == {0.0}:
            return True

        return False

    def _check_other_links(self, core_num, start_index, end_index):
        self.spectrum_props['is_free'] = True
        for i in range(len(self.spectrum_props['path_list']) - 1):
            link = (self.spectrum_props['path_list'][i], self.spectrum_props['path_list'][i + 1])
            rev_link = (self.spectrum_props['path_list'][i + 1], self.spectrum_props['path_list'][i])

            if not self._link_has_free_spectrum(link, rev_link, core_num, start_index, end_index):
                self.spectrum_props['is_free'] = False
                return

    # TODO: Break this up into at least 2 functions
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

    # TODO: Break this up into at least two methods
    # TODO: Might move to another file
    def _check_open_slots(self, open_slots_matrix: list, core_num: int):
        for tmp_arr in open_slots_matrix:
            # TODO: Slots needed has not been defined
            if len(tmp_arr) >= (self.spectrum_props['slots_needed'] + self.engine_props['guard_slots']):
                for start_index in tmp_arr:
                    if self.engine_props['allocation_method'] == 'last_fit':
                        end_index = (start_index - self.spectrum_props['slots_needed'] - self.engine_props[
                            'guard_slots']) + 1
                    else:
                        end_index = (start_index + self.spectrum_props['slots_needed'] + self.engine_props[
                            'guard_slots']) - 1
                    if end_index not in tmp_arr:
                        break

                    if len(self.spectrum_props['path_list']) > 2:
                        if self.engine_props['allocation_method'] == 'last_fit':
                            # Note that these are reversed since we search in decreasing order, but allocate in
                            # increasing order
                            self._check_other_links(core_num, end_index, start_index + self.engine_props['guard_slots'])
                        else:
                            self._check_other_links(core_num, start_index, end_index + self.engine_props['guard_slots'])

                    if self.spectrum_props['is_free'] is not False or len(self.spectrum_props['path_list']) <= 2:
                        # Since we use enumeration prior and set the matrix equal to one core, the "core_num" will
                        # always be zero even if our desired core index is different, is this lazy coding? Idek
                        # fixme no forced core here
                        # TODO: What?
                        if self.spectrum_props['forced_core'] is not None:
                            core_num = self.spectrum_props['forced_core']

                        # TODO: Can make this better
                        if self.engine_props['allocation_method'] == 'last_fit':
                            self.spectrum_props['start_slot'] = end_index
                            self.spectrum_props['end_slot'] = start_index + self.engine_props['guard_slots']
                        else:
                            self.spectrum_props['start_slot'] = start_index
                            self.spectrum_props['end_slot'] = end_index + self.engine_props['guard_slots']

                        self.spectrum_props['core_num'] = core_num
                        return True

        return False

    # TODO: Maybe two methods as well
    def _handle_first_last(self, flag: str = None):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method we'd like to use.
        :type flag: str
        """
        # TODO: Still need to implement forced core in other scripts
        # TODO: Need access to cores matrix here
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
            else:
                open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

            resp = self._check_open_slots(open_slots_matrix=open_slots_matrix, core_num=core_num)
            # We successfully found allocation on this core, no need to check the others
            if resp:
                return

    # TODO: Haven't technically used xt allocation, just make sure it runs
    # TODO: Definitely to helper script
    def _check_cores_channels(self):
        """
        For a given path, find the free channel intersections (between cores on that path) and the free slots for
        each link.

        :return: The free slots, free channel intersections, free slots intersections, and free channels.
        :rtype: dict
        """
        resp = {'free_slots': {}, 'free_channels': {}, 'slots_inters': {}, 'channel_inters': {}}

        for source_dest in zip(self.spectrum_props['path_list'], self.spectrum_props['path_list'][1:]):
            free_slots = self.find_free_slots(net_spec_db=self.sdn_props['net_spec_dict'], des_link=source_dest)
            free_channels = self.find_free_channels(net_spec_db=self.sdn_props['net_spec_dict'],
                                                    slots_needed=self.spectrum_props['slots_needed'],
                                                    des_link=source_dest)

            resp['free_slots'].update({source_dest: free_slots})
            resp['free_channels'].update({source_dest: free_channels})

            for core_num in resp['free_slots'][source_dest]:
                if core_num not in resp['slots_inters']:
                    resp['slots_inters'].update({core_num: set(resp['free_slots'][source_dest][core_num])})

                    resp['channel_inters'].update({core_num: resp['free_channels'][source_dest][core_num]})
                else:
                    intersection = resp['slots_inters'][core_num] & set(resp['free_slots'][source_dest][core_num])
                    resp['slots_inters'][core_num] = intersection
                    resp['channel_inters'][core_num] = [item for item in resp['channel_inters'][core_num] if
                                                        item in resp['free_channels'][source_dest][core_num]]

        return resp

    # TODO: probably to helper script
    def _find_best_core(self):
        """
        Finds the core with the least amount of overlapping super channels for previously allocated requests.

        :return: The core with the least amount of overlapping channels.
        :rtype: int
        """
        path_info = self._check_cores_channels()
        all_channels = self.get_channel_overlaps(path_info['channel_inters'],
                                                 path_info['free_slots'])
        sorted_cores = sorted(all_channels['other_channels'], key=lambda k: len(all_channels['other_channels'][k]))

        # TODO: Comment why
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return sorted_cores[0]

    def _xt_aware_allocation(self):
        """
        Cross-talk aware spectrum allocation. Attempts to allocate a request with the least amount of cross-talk
        interference on neighboring cores.

        :return: The information of the request if allocated, false otherwise.
        :rtype dict
        """
        if self.print_warn:
            warnings.warn('Method: xt_aware_allocation used in '
                          'spectrum_assignment that only supports 7 cores per fiber.')
        core = self._find_best_core()
        # Graph coloring for cores
        # TODO: Update des_core
        if core in [0, 2, 4, 6]:
            return self._handle_first_last(flag='first_fit')

        return self._handle_first_last(flag='last_fit')

    # TODO: Maybe two methods for this
    def _get_spectrum(self):
        # TODO: Only first and last fit supported for the moment
        # TODO: Is free should be the only flag that works
        if self.engine_props['allocation_method'] == 'best_fit':
            self._best_fit_allocation()
        elif self.engine_props['allocation_method'] in ('first_fit', 'last_fit'):
            self._handle_first_last()
        elif self.engine_props['allocation_method'] == 'cross_talk_aware':
            self._xt_aware_allocation()
        else:
            raise NotImplementedError(f"Expected first_fit or best_fit, got: {self.engine_props['allocation_method']}")

    # TODO: Still need to init more info here
    def _init_spectrum_info(self):
        self.spectrum_props = empty_props
        # TODO: Slots needed needs to be defined
        link_tuple = (self.spectrum_props['path_list'][0], self.spectrum_props['path_list'][1])
        rev_link_tuple = (self.spectrum_props['path_list'][1], self.spectrum_props['path_list'][0])
        self.spectrum_props['cores_matrix'] = self.sdn_props['net_spec_dict'][link_tuple]['cores_matrix']
        self.spectrum_props['rev_cores_matrix'] = self.sdn_props['net_spec_dict'][rev_link_tuple]['cores_matrix']

    def get_spectrum(self, mod_options: list):
        self._init_spectrum_info()
        # TODO: Need to define slots needed
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
                    # TODO: Need snr object here
                    # TODO: This will be much different after cleaning snr script
                    update_snr_obj(snr_obj=snr_obj, spectrum=spectrum, path=path, path_mod=path_mod,
                                   spectral_slots=self.engine_props['spectral_slots'], net_spec_db=net_spec_dict)
                    snr_check, xt_cost = handle_snr(check_snr=self.engine_props['check_snr'], snr_obj=snr_obj)
                    self.spectrum_props['xt_cost'] = xt_cost

                    # TODO: All of these will be returned as spectrum_props so we don't need them
                    # TODO: Make sure this series of statements and on are correct
                    if not snr_check:
                        return False, 'xt_threshold', xt_cost

                return

            self.spectrum_props['block_reason'] = 'congestion'
            return
