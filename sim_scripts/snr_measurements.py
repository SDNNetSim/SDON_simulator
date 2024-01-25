import math

import numpy as np
import networkx as nx

from arg_scripts.snr_args import empty_props


# fixme: Only works for seven cores
class SnrMeasurements:
    """
    Handles signal-to-noise ratio calculations for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: dict, spectrum_props: dict):
        self.snr_props = empty_props
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.spectrum_props = spectrum_props

        self.channels_list = None
        self.link_id = None
        self.num_slots = None

    def _calculate_sci_psd(self):
        """
        Calculates the self-phase power spectral density.

        :return: The self-phase power spectral density.
        :rtype: float
        """
        rho_param = (math.pi ** 2) * np.abs(self.link_dict['dispersion'])
        rho_param /= 2 * self.link_dict['attenuation']

        sci_psd = self.snr_props['center_psd'] ** 2
        sci_psd *= math.asinh(rho_param * (self.snr_props['bandwidth'] ** 2))
        return sci_psd

    def _update_link_xci(self, req_id: float, curr_link: np.ndarray, slot_index: int, curr_xci: float):
        """
        Given the spectrum contents, updates the link's cross-phase modulation noise.

        :return: The updated cross-phase modulation noise.
        :rtype: float
        """
        channel_bw = len(np.where(req_id == curr_link[self.spectrum_props['core_num']])[0])
        channel_bw *= self.engine_props['bw_per_slot']
        channel_freq = ((slot_index * self.engine_props['bw_per_slot']) + (channel_bw / 2)) * 10 ** 9
        channel_bw *= 10 ** 9
        channel_psd = self.engine_props['input_power'] / channel_bw

        if self.snr_props['center_freq'] != channel_freq:
            log_term = abs(self.snr_props['center_freq'] - channel_freq) + (channel_bw / 2)
            log_term /= abs(self.snr_props['center_freq'] - channel_freq) - (channel_bw / 2)
            calculated_xci = (channel_psd ** 2) * math.log(abs(log_term))
            new_xci = curr_xci + calculated_xci
        else:
            new_xci = curr_xci

        return new_xci

    def _calculate_xci(self, link_num: int):
        """
        Calculates the cross-phase modulation noise on a link for a single request.

        :return: The total cross-phase modulation noise on the link
        :rtype: float
        """
        self.channels_list = []
        # Cross-phase modulation noise
        xci_noise = 0
        for slot_index in range(self.engine_props['spectral_slots']):
            source = self.spectrum_props['path_list'][link_num]
            dest = self.spectrum_props['path_list'][link_num + 1]
            curr_link = self.sdn_props['net_spec_dict'][(source, dest)]['cores_matrix']
            req_id = curr_link[self.spectrum_props['core_num']][slot_index]

            # Spectrum is occupied
            if req_id > 0 and req_id not in self.channels_list:
                self.channels_list.append(req_id)
                xci_noise = self._update_link_xci(req_id=req_id, curr_link=curr_link,
                                                  slot_index=slot_index, curr_xci=xci_noise)

        return xci_noise

    # TODO: Change to num adjacent
    def _calculate_pxt(self, adjacent_cores: int):
        """
        Calculates the cross-talk noise power.

        :return: The cross-talk noise power normalized by the number of adjacent cores.
        :rtype: float
        """
        # A statistical mean of the cross-talk
        mean_xt = (2 * self.link_dict['bending_radius'] * self.link_dict['mode_coupling_co'] ** 2) / (
                self.link['propagation_const'] * self.link_dict['core_pitch'])
        # The cross-talk noise power
        # TODO: Should we use span or link length?
        power_xt = adjacent_cores * mean_xt * self.snr_props['length'] * 1e3 * self.engine_props['input_power']

        return power_xt

    # TODO: Change to num_adjacent
    def calculate_xt(self, adjacent_cores: int, link_length: int):
        """
        Calculates the cross-talk interference based on the number of adjacent cores.

        :return: The cross-talk normalized by the number of adjacent cores.
        :rtype: float
        """
        mean_xt = 3.78e-9
        resp_xt = (1 - math.exp(-2 * mean_xt * link_length * 1e3)) / (1 + math.exp(-2 * mean_xt * link_length * 1e3))

        return resp_xt * adjacent_cores

    def _handle_egn_model(self):
        """
        Calculates the power spectral density correction based on the EGN model.

        :return: The total power spectral density correction
        """
        # The harmonic number series
        hn_series = 0
        for i in range(1, math.ceil((len(self.channels_list) - 1) / 2) + 1):
            hn_series = hn_series + 1 / i

        # The effective span length
        eff_span_len = (1 - math.e ** (-2 * self.link_dict['attenuation'] * self.snr_props['length'] * 10 ** 3)) / (
                2 * self.link_dict['attenuation'])

        baud_rate = int(self.snr_props['req_bit_rate']) * 10 ** 9 / 2
        temp_coef = ((self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2) * (
                eff_span_len ** 2) * (self.snr_props['center_psd'] ** 3) * (self.snr_props['bandwidth'] ** 2)) / (
                            (baud_rate ** 2) * math.pi * self.link_dict['dispersion'] * (
                            self.snr_props['length'] * 10 ** 3))

        # The PSD correction term
        psd_correction = (80 / 81) * self.engine_props['phi'][self.spectrum_props['modulation']] * temp_coef * hn_series

        return psd_correction

    def _calculate_psd_nli(self):
        """
        Calculates the power spectral density non-linear interference for a link.

        :return: The total power spectral density non-linear interference
        :rtype float
        """
        # Determine if we're using the GN or EGN model
        if self.engine_props['egn_model']:
            psd_correction = self._handle_egn_model()
            psd_nli = ((self.snr_props['sci_psd'] + self.snr_props['xci_psd']) * self.snr_props['mu_param'] *
                       self.snr_props['center_psd']) - psd_correction
        else:
            psd_nli = (self.snr_props['sci_psd'] + self.snr_props['xci_psd']) * self.snr_props['mu_param'] * \
                      self.snr_props['center_psd']

        return psd_nli

    # TODO: Change to link num
    def _update_link_params(self, link: int):
        """
        Updates needed parameters for each link used for calculating SNR or XT.
        """
        self.snr_props['mu_param'] = (3 * (
                self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2)) / (
                                             2 * math.pi * self.link_dict['attenuation'] * np.abs(
                                         self.link_dict['dispersion']))
        self.snr_props['sci_psd'] = self._calculate_sci_psd()
        self.snr_props['xci_psd'] = self._calculate_xci(link=link)
        # TODO Add support for self.engine_props['topology_info']['links'][link_id]['fiber']['nsp']
        # self.snr_props['nsp'] = 1.8

        self.snr_props['length'] = self.engine_props['topology_info']['links'][self.link_id]['span_length']
        self.snr_props['num_span'] = self.engine_props['topology_info']['links'][self.link_id]['length'] / \
                                     self.snr_props['length']

    def _init_center_vars(self):
        """
        Updates variables for the center frequency, bandwidth, and PSD for the current request.
        """
        self.snr_props['center_freq'] = ((self.spectrum_props['start_slot'] * self.engine_props['bw_per_slot']) + (
                (self.num_slots * self.engine_props['bw_per_slot']) / 2)) * 10 ** 9

        self.snr_props['bandwidth'] = self.num_slots * self.engine_props['bw_per_slot'] * 10 ** 9
        self.snr_props['center_psd'] = self.engine_props['input_power'] / self.snr_props['bandwidth']

    def check_snr(self):
        """
        Determines whether the SNR threshold can be met for a single request.

        :return: Whether the SNR threshold can be met.
        :rtype: bool
        """
        snr = 0
        self._init_center_vars()
        for link in range(0, len(self.spectrum_props['path_list']) - 1):
            self.link_id = self.sdn_props['net_spec_dict'][
                (self.spectrum_props['path_list'][link], self.spectrum_props['path_list'][link + 1])]['link_num']

            self.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            self._update_link_params(link=link)

            psd_nli = self._calculate_psd_nli()
            psd_ase = (self.snr_props['plank'] * self.snr_props['light_frequency'] * self.snr_props['nsp']) * (
                    math.exp(self.link_dict['attenuation'] * self.snr_props['length'] * 10 ** 3) - 1)
            if self.engine_props['xt_noise']:
                p_xt = self._calculate_pxt(adjacent_cores=None)
            else:
                p_xt = 0

            snr += (1 / ((self.snr_props['center_psd'] * self.snr_props['bandwidth']) / (
                    ((psd_ase + psd_nli) * self.snr_props['bandwidth'] + p_xt) * self.snr_props['num_span'])))

        snr = 10 * math.log10(1 / snr)

        resp = snr > self.snr_props['req_snr']
        return resp

    def check_snr_xt(self):
        """
        Determines whether the SNR threshold can be met for a single request.

        :return: Whether the SNR threshold can be met.
        :rtype: bool
        """
        snr = 0
        self._init_center_vars()
        for link in range(0, len(self.spectrum_props['path_list']) - 1):
            self.link_id = self.sdn_props['net_spec_dict'][
                (self.spectrum_props['path_list'][link], self.spectrum_props['path_list'][link + 1])]['link_num']

            self.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            # self.update_link_constants()
            self._update_link_params(link=link)
            psd_ase = (self.snr_props['plank'] * self.snr_props['light_frequency'] * self.snr_props['nsp']) * (
                    math.exp(self.link_dict['attenuation'] * self.snr_props['length'] * 10 ** 3) - 1)
            if self.engine_props['xt_noise']:
                p_xt = self._calculate_pxt(adjacent_cores=None)
            else:
                p_xt = 0

            snr += (1 / ((self.snr_props['center_psd'] * self.snr_props['bandwidth']) / (
                    (psd_ase * self.snr_props['bandwidth'] + p_xt) * self.snr_props['num_span'])))

        snr = 10 * math.log10(1 / snr)

        resp = snr > self.snr_props['req_snr']
        return resp

    # TODO: Change to link tuple
    def check_adjacent_cores(self, link_nodes: tuple):
        """
        Given a link, finds the number of cores which have overlapping channels on a fiber.

        :param link_nodes: The source and destination nodes.
        :type link_nodes: tuple

        :return: The number of adjacent cores that have overlapping channels.
        """
        resp = 0
        if self.spectrum_props['core_num'] != 6:
            # The neighboring core directly before the currently selected core
            before = 5 if self.spectrum_props['core_num'] == 0 else self.spectrum_props['core_num'] - 1
            # The neighboring core directly after the currently selected core
            after = 0 if self.spectrum_props['core_num'] == 5 else self.spectrum_props['core_num'] + 1
            adjacent_cores = [before, after, 6]
        else:
            adjacent_cores = list(range(6))

        for curr_slot in range(self.spectrum_props['start_slot'], self.spectrum_props['end_slot']):
            overlapped = 0
            for core_num in adjacent_cores:
                core_contents = self.sdn_props['net_spec_dict'][link_nodes]['cores_matrix'][core_num][curr_slot]
                if core_contents > 0.0:
                    overlapped += 1

            # Determine which slot has the maximum number of overlapping channels
            if overlapped > resp:
                resp = overlapped

        return resp

    def find_worst_xt(self, flag: str):
        """
        Finds the worst possible cross-talk.

        :param flag: Determines which type of cross-talk is being considered.
        :type flag: str

        :return: The maximum length of the link found and the cross-talk calculated.
        :rtype: tuple
        """
        if flag == 'intra_core':
            edge_lengths = nx.get_edge_attributes(self.engine_props['topology'], 'length')
            max_link = max(edge_lengths, key=edge_lengths.get, default=None)
            self.link_id = self.sdn_props['net_spec_dict'][max_link]['link_num']
            max_length = edge_lengths.get(max_link, 0.0)
            self.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            # self.update_link_constants()
            resp = self.calculate_xt(adjacent_cores=6, link_length=max_length)
            resp = 10 * math.log10(resp)
        else:
            raise NotImplementedError

        return resp, max_length

    def check_xt(self):
        """
        Checks the amount of cross-talk interference on a single request.

        :return: Whether the cross-talk interference threshold can be met
        :rtype: bool
        """
        cross_talk = 0

        self._init_center_vars()
        for link in range(0, len(self.spectrum_props['path_list']) - 1):
            link_nodes = (self.spectrum_props['path_list'][link], self.spectrum_props['path_list'][link + 1])
            self.link_id = self.sdn_props['net_spec_dict'][link_nodes]['link_num']
            link_length = self.engine_props['topology_info']['links'][self.link_id]['length']
            self.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            # self.update_link_constants()
            self._update_link_params(link=link)

            adjacent_cores = self.check_adjacent_cores(link_nodes=link_nodes)
            cross_talk += self.calculate_xt(adjacent_cores=adjacent_cores, link_length=link_length)

        if cross_talk == 0:
            resp = True
        else:
            cross_talk = 10 * math.log10(cross_talk)
            resp = cross_talk < self.engine_props['requested_xt'][self.spectrum_props['modulation']]

        return resp, cross_talk

    def handle_snr(self):
        self.num_slots = self.spectrum_props['end_slot'] - self.spectrum_props['start_slot'] + 1
        # TODO: Changed from snr_calculation_nli to snr_calc_nli
        # TODO: This param should NOT be called check_snr
        if self.engine_props['check_snr'] == "snr_calc_nli":
            snr_check, xt_cost = self.check_snr()
        elif self.engine_props['check_snr'] == "xt_calculation":
            snr_check, xt_cost = self.check_xt()
            # TODO: Changed from snr_calculation_xt to snr_calc_xt
        elif self.engine_props['check_snr'] == "snr_calc_xt":
            snr_check, xt_cost = self.check_snr_xt()
        else:
            raise NotImplementedError(f"Unexpected check_snr flag got: {self.engine_props['check_snr']}")

        return snr_check, xt_cost
