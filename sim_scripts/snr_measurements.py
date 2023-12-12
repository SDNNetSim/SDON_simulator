import math

import numpy as np
import networkx as nx


class SnrMeasurements:
    """
    Calculates SNR for a given request.
    """

    def __init__(self, properties: dict):
        """
        Initializes the SnrMeasurements class.

        :param properties: Contains various simulation properties.
        :type properties: dict
        """
        self.snr_props = properties

        self.light_frequency = 1.9341 * 10 ** 14
        self.plank = 6.62607004e-34
        self.req_bit_rate = 12.5
        self.req_snr = 8.5
        self.requests_status = {}
        self.path = list()
        self.spectrum = dict()
        self.net_spec_db = dict()

        self.attenuation = None
        self.dispersion = None
        self.bend_radius = None
        self.coupling_coeff = None
        self.prop_const = None
        self.core_pitch = None

        # The center frequency
        self.center_freq = None
        # The power spectral density for the center channel
        self.center_psd = None
        # The current requests bandwidth
        self.bandwidth = None

        # Used as a parameter for the GN model
        self.mu_param = None
        # The self-phase power spectral density
        self.sci_psd = None
        # Cross-phase modulation power spectral density
        self.xci_psd = None
        self.visited_channels = None
        self.length = None
        self.nsp = None
        self.num_span = None

        self.link_id = None
        self.link = None
        self.path_mod = None
        self.assigned_slots = None
        self.baud_rates = None
        self.baud_rates = None

    def _calculate_sci_psd(self):
        """
        Calculates the self-phase power spectral density.

        :return: The self-phase power spectral density.
        :rtype: float
        """
        rho_param = ((math.pi ** 2) * np.abs(self.dispersion)) / (2 * self.attenuation)
        sci_psd = (self.center_psd ** 2) * math.asinh(rho_param * (self.bandwidth ** 2))
        return sci_psd

    # TODO: I believe repeat code, calculate mci function in routing.py
    def _update_link_xci(self, spectrum_contents: float, curr_link: np.ndarray, slot_index: int, curr_xci: float):
        """
        Given the spectrum contents, updates the link's cross-phase modulation noise.

        :param spectrum_contents: The request number if the spectrum is occupied (zero otherwise).
        :type spectrum_contents: float

        :param curr_link: The current link's contents.
        :type curr_link: np.ndarray

        :param slot_index: The current slot index of the channel we're searching on.
        :type slot_index: int

        :param curr_xci: The total cross-phase modulation noise calculated thus far.
        :type curr_xci: float

        :return: The updated cross-phase modulation noise.
        :rtype: float
        """
        num_slots = len(np.where(spectrum_contents == curr_link[self.spectrum['core_num']])[0]) * self.snr_props[
            'bw_per_slot']
        channel_freq = ((slot_index * self.snr_props['bw_per_slot']) + (num_slots / 2)) * 10 ** 9

        channel_bw = num_slots * 10 ** 9
        channel_psd = self.snr_props['input_power'] / channel_bw
        if self.center_freq != channel_freq:
            new_xci = curr_xci + ((channel_psd ** 2) * math.log(
                abs((abs(self.center_freq - channel_freq) + (channel_bw / 2)) / (
                        abs(self.center_freq - channel_freq) - (channel_bw / 2)))))
        else:
            new_xci = curr_xci

        return new_xci

    # TODO: I believe I've implemented something similar in routing (calculate MCI on my branch)
    # TODO: That doesn't have multi-core support I believe, and, probably better to move here instead
    def _calculate_xci(self, link: int):
        """
        Calculates the cross-phase modulation noise on a link for a single request.

        :param link: The current link number of the given path we're on.
        :type link: int

        :return: The total cross-phase modulation noise on the link
        :rtype: float
        """
        self.visited_channels = []
        # Cross-phase modulation noise
        xci_noise = 0
        for slot_index in range(self.snr_props['spectral_slots']):
            curr_link = self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix']
            spectrum_contents = curr_link[self.spectrum['core_num']][slot_index]

            # Spectrum is occupied
            if spectrum_contents > 0 and spectrum_contents not in self.visited_channels:
                self.visited_channels.append(spectrum_contents)
                xci_noise = self._update_link_xci(spectrum_contents=spectrum_contents, curr_link=curr_link,
                                                  slot_index=slot_index, curr_xci=xci_noise)

        return xci_noise

    def _calculate_pxt(self, adjacent_cores: int):
        """
        Calculates the cross-talk noise power.

        :param adjacent_cores: The number of adjacent cores to the channel.
        :type adjacent_cores: int

        :return: The cross-talk noise power normalized by the number of adjacent cores.
        :rtype: float
        """
        # A statistical mean of the cross-talk
        mean_xt = (2 * self.bend_radius * self.coupling_coeff ** 2) / (self.prop_const * self.core_pitch)
        # The cross-talk noise power
        # TODO: Should we use span or link length?
        power_xt = adjacent_cores * mean_xt * self.length * 1e3 * self.snr_props['input_power']

        return power_xt

    def calculate_xt(self, adjacent_cores: int, link_length: int):
        """
        Calculates the cross-talk interference based on the number of adjacent cores & and the length of the link.

        :param adjacent_cores: The number of adjacent cores to the channel.
        :type adjacent_cores: int

        :param link_length: The current length of the link.
        :type link_length: int

        :return: The cross-talk normalized by the number of adjacent cores.
        :rtype: float
        """
        mean_xt = (2 * self.bend_radius * (self.coupling_coeff ** 2)) / (self.prop_const * self.core_pitch)
        resp_xt = (1 - math.exp(-2 * mean_xt * link_length * 1e3)) / (1 + math.exp(-2 * mean_xt * link_length * 1e3))

        return resp_xt * adjacent_cores

    def _handle_egn_model(self):
        """
        Calculates the power spectral density correction based on the EGN model.

        :return: The total power spectral density correction
        :rtype: Union[float,int]
        """
        # The harmonic number series
        hn_series = 0
        for i in range(1, math.ceil((len(self.visited_channels) - 1) / 2) + 1):
            hn_series = hn_series + 1 / i

        # The effective span length
        eff_span_len = (1 - math.e ** (-2 * self.attenuation * self.length * 10 ** 3)) / (2 * self.attenuation)

        baud_rate = int(self.req_bit_rate) * 10 ** 9 / 2
        temp_coef = ((self.snr_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2) * (
                eff_span_len ** 2) * (self.center_psd ** 3) * (self.bandwidth ** 2)) / (
                            (baud_rate ** 2) * math.pi * self.dispersion * (self.length * 10 ** 3))

        # The PSD correction term
        psd_correction = (80 / 81) * self.snr_props['phi'][self.path_mod] * temp_coef * hn_series

        return psd_correction

    def _calculate_psd_nli(self):
        """
        Calculates the power spectral density non-linear interference for a link.

        :return: The total power spectral density non-linear interference
        :rtype: float
        """
        # Determine if we're using the GN or EGN model
        if self.snr_props['egn_model']:
            psd_correction = self._handle_egn_model()
            psd_nli = ((self.sci_psd + self.xci_psd) * self.mu_param * self.center_psd) - psd_correction
        else:
            psd_nli = (self.sci_psd + self.xci_psd) * self.mu_param * self.center_psd

        return psd_nli

    def _update_link_params(self, link: int):
        """
        Updates needed parameters for each link used for calculating SNR or XT.

        :param link: The current link number of the path we are on.
        :type link: int

        :return: None
        """
        self.mu_param = (3 * (
                self.snr_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2)) / (
                                2 * math.pi * self.attenuation * np.abs(self.dispersion))
        self.sci_psd = self._calculate_sci_psd()
        self.xci_psd = self._calculate_xci(link=link)
        # TODO Add support for self.snr_props['topology_info']['links'][link_id]['fiber']['nsp']
        self.nsp = 1.8

        self.length = self.snr_props['topology_info']['links'][self.link_id]['span_length']
        self.num_span = self.snr_props['topology_info']['links'][self.link_id]['length'] / self.length

    def update_link_constants(self):
        """
        Updates non-linear impairment parameters that will remain constant for future calculations.

        :return: None
        """
        self.link = self.snr_props['topology_info']['links'][self.link_id]['fiber']
        self.attenuation = self.link['attenuation']
        self.dispersion = self.link['dispersion']
        self.bend_radius = self.link['bending_radius']
        self.coupling_coeff = self.link['mode_coupling_co']
        self.prop_const = self.link['propagation_const']
        self.core_pitch = self.link['core_pitch']

    def _init_center_vars(self):
        """
        Updates variables for the center frequency, bandwidth, and PSD for the current request.

        :return: None
        """
        self.center_freq = ((self.spectrum['start_slot'] * self.snr_props['bw_per_slot']) + (
                (self.assigned_slots * self.snr_props['bw_per_slot']) / 2)) * 10 ** 9
        self.bandwidth = self.assigned_slots * self.snr_props['bw_per_slot'] * 10 ** 9
        self.center_psd = self.snr_props['input_power'] / self.bandwidth

    def check_snr(self):
        """
        Determines whether the Signal-to-Noise Ratio (SNR) threshold can be met for a single request.

        :return: Whether the SNR threshold can be met.
        :rtype: bool
        """
        snr = 0
        self._init_center_vars()
        for link in range(0, len(self.path) - 1):
            self.link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']

            self.update_link_constants()
            self._update_link_params(link=link)

            psd_nli = self._calculate_psd_nli()
            psd_ase = (self.plank * self.light_frequency * self.nsp) * (
                    math.exp(self.attenuation * self.length * 10 ** 3) - 1)
            if self.snr_props['xt_noise']:
                p_xt = self._calculate_pxt(adjacent_cores=None)
            else:
                p_xt = 0

            snr += (1 / ((self.center_psd * self.bandwidth) / (
                    ((psd_ase + psd_nli) * self.bandwidth + p_xt) * self.num_span)))

        snr = 10 * math.log10(1 / snr)

        resp = snr > self.req_snr
        return resp

    def check_snr_xt(self):
        """
        Determines whether the Signal-to-Noise (SNR) threshold can be met for a single request under applied cross-talk (xt).

        :return: Whether the SNR threshold can be met under applied crosstalk.
        :rtype: bool
        """
        snr = 0
        self._init_center_vars()
        for link in range(0, len(self.path) - 1):
            self.link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']

            self.update_link_constants()
            self._update_link_params(link=link)
            psd_ase = (self.plank * self.light_frequency * self.nsp) * (
                    math.exp(self.attenuation * self.length * 10 ** 3) - 1)
            if self.snr_props['xt_noise']:
                p_xt = self._calculate_pxt(adjacent_cores=None)
            else:
                p_xt = 0

            snr += (1 / ((self.center_psd * self.bandwidth) / (
                    (psd_ase * self.bandwidth + p_xt) * self.num_span)))

        snr = 10 * math.log10(1 / snr)

        resp = snr > self.req_snr
        return resp

    def check_adjacent_cores(self, link_nodes: tuple):
        """
        Given a link, finds the number of cores which have overlapping channels (adjacency) on a fiber.

        :param link_nodes: The source and destination nodes.
        :type link_nodes: tuple

        :return: The number of adjacent cores that have overlapping channels.
        :rtype: int
        """
        resp = 0
        if self.spectrum['core_num'] != 6:
            # The neighboring core directly before the currently selected core
            before = 5 if self.spectrum['core_num'] == 0 else self.spectrum['core_num'] - 1
            # The neighboring core directly after the currently selected core
            after = 0 if self.spectrum['core_num'] == 5 else self.spectrum['core_num'] + 1
            adjacent_cores = [before, after, 6]
        else:
            adjacent_cores = list(range(6))

        for curr_slot in range(self.spectrum['start_slot'], self.spectrum['end_slot']):
            overlapped = 0
            for core_num in adjacent_cores:
                core_contents = self.net_spec_db[link_nodes]['cores_matrix'][core_num][curr_slot]
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
            edge_lengths = nx.get_edge_attributes(self.snr_props['topology'], 'length')
            max_link = max(edge_lengths, key=edge_lengths.get, default=None)
            self.link_id = self.net_spec_db[max_link]['link_num']
            max_length = edge_lengths.get(max_link, 0.0)
            self.update_link_constants()
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
        for link in range(0, len(self.path) - 1):
            link_nodes = (self.path[link], self.path[link + 1])
            self.link_id = self.net_spec_db[link_nodes]['link_num']
            link_length = self.snr_props['topology_info']['links'][self.link_id]['length']
            self.update_link_constants()
            self._update_link_params(link=link)

            adjacent_cores = self.check_adjacent_cores(link_nodes=link_nodes)
            cross_talk += self.calculate_xt(adjacent_cores=adjacent_cores, link_length=link_length)

        if cross_talk == 0:
            resp = True
        else:
            cross_talk = 10 * math.log10(cross_talk)
            resp = cross_talk < self.snr_props['requested_xt'][self.path_mod]

        return resp, cross_talk
