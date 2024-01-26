from helper_scripts.sim_helpers import find_free_channels, find_free_slots, get_channel_overlaps


# TODO: I don't think we need to return constructor variables
class SpectrumHelpers:
    def __init__(self, engine_props: dict, sdn_props: dict, spectrum_props: dict):
        self.engine_props = engine_props
        self.spectrum_props = spectrum_props
        self.sdn_props = sdn_props

        self.start_index = None
        self.end_index = None
        self.core_num = None

    def _check_free_spectrum(self, link_tuple: tuple, rev_link_tuple: tuple):
        core_arr = self.sdn_props['net_spec_dict'][link_tuple]['cores_matrix'][self.core_num]
        spectrum_set = core_arr[self.start_index:self.end_index]
        rev_core_arr = self.sdn_props['net_spec_dict'][rev_link_tuple]['cores_matrix'][self.core_num]
        rev_spectrum_set = rev_core_arr[self.start_index:self.end_index]

        if set(spectrum_set) == {0.0} and set(rev_spectrum_set) == {0.0}:
            return True

        return False

    def check_other_links(self):
        self.spectrum_props['is_free'] = True
        for node in range(len(self.spectrum_props['path_list']) - 1):
            link_tuple = (self.spectrum_props['path_list'][node], self.spectrum_props['path_list'][node + 1])
            rev_link_tuple = (self.spectrum_props['path_list'][node + 1], self.spectrum_props['path_list'][node])

            if not self._check_free_spectrum(link_tuple=link_tuple, rev_link_tuple=rev_link_tuple):
                self.spectrum_props['is_free'] = False
                return

    def _update_spec_props(self):
        if self.spectrum_props['forced_core'] is not None:
            self.core_num = self.spectrum_props['forced_core']

        if self.engine_props['allocation_method'] == 'last_fit':
            self.spectrum_props['start_slot'] = self.end_index
            self.spectrum_props['end_slot'] = self.start_index + self.engine_props['guard_slots']
        else:
            self.spectrum_props['start_slot'] = self.start_index
            self.spectrum_props['end_slot'] = self.end_index + self.engine_props['guard_slots']

        self.spectrum_props['core_num'] = self.core_num
        return self.spectrum_props

    def check_super_channels(self, open_slots_matrix: list):
        """
        Given a matrix of available super-channels, find one that can allocate the current request.

        :param sdn_props: Properties of the SDN controller.
        :param spectrum_props: Properties of the spectrum assignment class.
        :param engine_props: Properties of the engine class.
        :param open_slots_matrix: A matrix where each entry is an available super-channel's indexes.
        :param core_num: The core number which is currently being checked.
        :return: If the request can be successfully allocated.
        :rtype: bool
        """
        for super_channel in open_slots_matrix:
            if len(super_channel) >= (self.spectrum_props['slots_needed'] + self.engine_props['guard_slots']):
                for start_index in super_channel:
                    if self.engine_props['allocation_method'] == 'last_fit':
                        end_index = (start_index - self.spectrum_props['slots_needed'] - self.engine_props[
                            'guard_slots']) + 1
                    else:
                        end_index = (start_index + self.spectrum_props['slots_needed'] + self.engine_props[
                            'guard_slots']) - 1
                    if end_index not in super_channel:
                        break
                    else:
                        self.spectrum_props['is_free'] = True

                    if len(self.spectrum_props['path_list']) > 2:
                        self.check_other_links()

                    if self.spectrum_props['is_free'] is not False or len(self.spectrum_props['path_list']) <= 2:
                        self._update_spec_props()
                        return True

        return False

    # TODO: This should be two functions, also does NOT find overlapping
    def find_link_inters(self):
        """
        Finds all slots and super-channels that have overlapping allocated requests. Also find free channels and slots.

        :param sdn_props: The properties of the SDN class.
        :param spectrum_props: The properties of the spectrum assignment class.
        :return: The free slots and super-channels along with intersecting slots and super-channels.
        :rtype: dict
        """
        resp = {'free_slots_dict': {}, 'free_channels_dict': {}, 'slots_inters_dict': {}, 'channel_inters_dict': {}}

        for source_dest in zip(self.spectrum_props['path_list'], self.spectrum_props['path_list'][1:]):
            free_slots = find_free_slots(net_spec_dict=self.sdn_props['net_spec_dict'], link_tuple=source_dest)
            free_channels = find_free_channels(net_spec_dict=self.sdn_props['net_spec_dict'],
                                               slots_needed=self.spectrum_props['slots_needed'],
                                               link_tuple=source_dest)

            resp['free_slots_dict'].update({source_dest: free_slots})
            resp['free_channels_dict'].update({source_dest: free_channels})

            for core_num in resp['free_slots_dict'][source_dest]:
                if core_num not in resp['slots_inters_dict']:
                    resp['slots_inters_dict'].update({core_num: set(resp['free_slots_dict'][source_dest][core_num])})

                    resp['channel_inters_dict'].update({core_num: resp['free_channels_dict'][source_dest][core_num]})
                else:
                    intersection = resp['slots_inters_dict'][core_num] & set(
                        resp['free_slots_dict'][source_dest][core_num])
                    resp['slots_inters_dict'][core_num] = intersection
                    resp['channel_inters_dict'][core_num] = [item for item in resp['channel_inters_dict'][core_num] if
                                                             item in resp['free_channels_dict'][source_dest][core_num]]

        return resp

    def find_best_core(self):
        """
        Finds the core with the least amount of overlapping super channels.

        :param sdn_props: Properties of the SDN controller.
        :param spectrum_props: Properties of the spectrum assignment class.
        :return: The core with the least amount of overlapping channels.
        :rtype: int
        """
        path_info = self.find_link_inters()
        all_channels = get_channel_overlaps(path_info['channel_inters_dict'],
                                            path_info['free_slots_dict'])
        sorted_cores = sorted(all_channels['non_over_dict'], key=lambda k: len(all_channels['non_over_dict'][k]))

        # TODO: Comment why
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return sorted_cores[0]
