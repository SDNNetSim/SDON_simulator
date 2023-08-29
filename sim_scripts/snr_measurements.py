import math

import numpy as np


class SnrMeasurements:
    """
    Calculates SNR for a given request.
    """

    # TODO: Might be a good idea to reference all constants to a paper or something
    # TODO: Move as much of this to the configuration file as possible
    def __init__(self,
                 path=None,
                 path_mod=None,
                 spectrum=None,
                 assigned_slots=None,
                 req_bit_rate=12.5,
                 freq_spacing=12.5,
                 input_power=10 ** -3,
                 spectral_slots=None,
                 req_snr=8.5,
                 net_spec_db=None,
                 topology_info=None,
                 # TODO: Delete if always constant
                 req_status=None,
                 phi=None,
                 guard_slots=0,
                 baud_rates=None,
                 # TODO: Either need a comment or better name
                 egn=False,
                 xt_noise=False,
                 bi_directional=True,
                 requested_xt=-30):

        # TODO: If these values will not change, it's best to take them out of the params of the constructor
        # TODO: Comments for a lot of these
        self.path = path
        self.spectrum = spectrum
        self.assigned_slots = assigned_slots
        self.req_bit_rate = req_bit_rate
        self.path_mod = path_mod
        self.guard_slots = guard_slots
        self.freq_spacing = freq_spacing
        self.input_power = input_power
        self.spectral_slots = spectral_slots
        self.req_snr = req_snr
        self.net_spec_db = net_spec_db
        self.topology_info = topology_info
        self.phi = {'QPSK': 1, '16-QAM': 0.68, '64-QAM': 0.6190476190476191}
        self.egn = egn
        self.requests_status = {}
        self.baud_rates = baud_rates
        self.bi_directional = bi_directional
        self.xt_noise = xt_noise
        self.plank = 6.62607004e-34
        self.requested_xt = requested_xt

        # TODO: This may have to get updated in another method since we don't create a new object every time
        self.response = {'SNR': None}

    # TODO: Repeat code (Move to useful functions)
    def _find_taken_channels(self, link_num: tuple):
        """
        Finds the number of taken channels on any given link.

        :param link_num: The link number to search for channels on.
        :type link_num: int

        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
        channels = []
        curr_channel = []
        link = self.net_spec_db[link_num]['cores_matrix'][0]

        for value in link:
            if value > 0:
                curr_channel.append(value)
            elif value < 0 and curr_channel:
                channels.append(curr_channel)
                curr_channel = []

        if curr_channel:
            channels.append(curr_channel)

        return channels

    def _SCI_calculator(self, link_id, PSDi, BW):
        Rho = ((math.pi ** 2) * np.abs(self.topology_info['links'][link_id]['fiber']['dispersion'])) / (
                2 * self.topology_info['links'][link_id]['fiber']['attenuation'])
        G_SCI = (PSDi ** 2) * math.asinh(Rho * (BW ** 2))
        return G_SCI

    def _XCI_calculator(self, Fi, link):
        visited_channel = []
        MCI = 0
        for w in range(self.spectral_slots):
            if self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix'][self.spectrum['core_num']][
                w] > 0:
                if self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix'][self.spectrum['core_num']][
                    w] in visited_channel:
                    continue
                else:
                    visited_channel.append(self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix'][
                                               self.spectrum['core_num']][w])
                BW_J = len(np.where(
                    self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix'][self.spectrum['core_num']][
                        w] == self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix'][
                        self.spectrum['core_num']])[0]) * self.freq_spacing
                Fj = ((w * self.freq_spacing) + ((BW_J) / 2)) * 10 ** 9
                BWj = BW_J * 10 ** 9
                PSDj = self.input_power / BWj
                if Fi != Fj:
                    MCI = MCI + ((PSDj ** 2) * math.log(abs((abs(Fi - Fj) + (BWj / 2)) / (abs(Fi - Fj) - (BWj / 2)))))
        return MCI, visited_channel

    def _PXT_calculator(self, link_id, length, no_adjacent_core=6):
        XT_eta = (2 * self.topology_info['links'][link_id]['fiber']["bending_radius"] *
                  self.topology_info['links'][link_id]['fiber']["mode_coupling_co"] ** 2) / (
                         self.topology_info['links'][link_id]['fiber']["propagation_const"] *
                         self.topology_info['links'][link_id]['fiber']["core_pitch"])
        P_XT = no_adjacent_core * XT_eta * length * 1e3 * self.input_power
        return P_XT

    def _XT_calculator(self, link_id, length, no_adjacent_core=6):
        XT_eta = (2 * self.topology_info['links'][link_id]['fiber']["bending_radius"] * (
                self.topology_info['links'][link_id]['fiber']["mode_coupling_co"] ** 2)) / (
                         self.topology_info['links'][link_id]['fiber']["propagation_const"] *
                         self.topology_info['links'][link_id]['fiber']["core_pitch"])
        XT_calc = (1 - math.exp(-2 * XT_eta * length * 1e3)) / (1 + math.exp(-2 * XT_eta * length * 1e3))
        return XT_calc * no_adjacent_core

    # TODO: Always returning False
    def SNR_check_NLI_ASE_XT(self):

        # TODO: Move to a constants file
        light_frequncy = (1.9341 * 10 ** 14)

        Fi = ((self.spectrum['start_slot'] * self.freq_spacing) + (
                (self.assigned_slots * self.freq_spacing) / 2)) * 10 ** 9
        BW = self.assigned_slots * self.freq_spacing * 10 ** 9
        PSDi = self.input_power / BW
        PSD_NLI = 0
        PSD_corr = 0
        SNR = 0
        for link in range(0, len(self.path) - 1):
            MCI = 0
            Num_span = 0
            # visited_channel = []
            link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']
            Mio = (3 * (self.topology_info['links'][link_id]['fiber']['non_linearity'] ** 2)) / (
                    2 * math.pi * self.topology_info['links'][link_id]['fiber']['attenuation'] * np.abs(
                self.topology_info['links'][link_id]['fiber']['dispersion']))
            G_SCI = self._SCI_calculator(link_id, PSDi, BW)
            G_XCI, visited_channel = self._XCI_calculator(Fi, link)
            length = self.topology_info['links'][link_id]['span_length']
            nsp = 1.8  # TODO self.topology_info['links'][link_id]['fiber']['nsp']
            Num_span = self.topology_info['links'][link_id]['length'] / length
            if self.egn:
                hn = 0
                for i in range(1, math.ceil((len(visited_channel) - 1) / 2) + 1):
                    hn = hn + 1 / i
                effective_L = (1 - math.e ** (-2 * self.topology_info['links'][link_id]['fiber'][
                    'attenuation'] * length * 10 ** 3)) / (
                                      2 * self.topology_info['links'][link_id]['fiber']['attenuation'])
                baud_rate = int(self.req_bit_rate) * 10 ** 9 / 2  # self.baud_rates[self.modulation_format]
                temp_coef = ((self.topology_info['links'][link_id]['fiber']['non_linearity'] ** 2) * (
                        effective_L ** 2) * (PSDi ** 3) * (BW ** 2)) / (
                                    (baud_rate ** 2) * math.pi * self.topology_info['links'][link_id]['fiber'][
                                'dispersion'] * (length * 10 ** 3))
                PSD_corr = (80 / 81) * self.phi[self.path_mod] * temp_coef * hn

            PSD_ASE = 0
            if self.egn:
                PSD_NLI = (((G_SCI + G_XCI) * Mio * PSDi)) - PSD_corr
            else:
                PSD_NLI = (((G_SCI + G_XCI) * Mio * PSDi))
            PSD_ASE = (self.plank * light_frequncy * nsp) * (math.exp(
                self.topology_info['links'][link_id]['fiber']['attenuation'] * length * 10 ** 3) - 1)
            P_XT = self._PXT_calculator(link_id, length)
            SNR += (1 / ((PSDi * BW) / (((PSD_ASE + PSD_NLI) * BW + P_XT) * Num_span)))

        SNR = 10 * math.log10(1 / SNR)
        return True if SNR > self.req_snr else False

    def XT_check(self):
        XT = 0
        for link in range(0, len(self.path) - 1):
            link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']
            length = self.topology_info['links'][link_id]['span_length']
            Num_span = self.topology_info['links'][link_id]['length'] / length
            XT += self._XT_calculator(link_id, length) * Num_span
        XT = 10 * math.log10(XT)
        return True if XT < self.requested_xt else False
