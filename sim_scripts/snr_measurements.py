import math

import numpy as np


class SnrMeasurements:
    """
    Calculates SNR for a given request.
    """

    # TODO: Might be a good idea to reference all constants to a paper or something
    # TODO: Move as much of this to the configuration file as possible
    # TODO: Delete params from arguments if they're always constant
    # TODO: Better naming conventions (overview of the script)
    def __init__(self, path=None, path_mod=None, spectrum=None, assigned_slots=None, req_bit_rate=12.5,
                 freq_spacing=12.5, input_power=10 ** -3, spectral_slots=None, req_snr=8.5, net_spec_db=None,
                 topology_info=None, req_status=None, phi=None, guard_slots=0, baud_rates=None, egn=False,
                 xt_noise=False, bi_directional=True, requested_xt=-30):

        # TODO: If these values will not change, it's best to take them out of the params of the constructor
        # TODO: Comments for all of these
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

        # TODO: Update this every time
        self.attenuation = None
        self.dispersion = None
        self.bend_radius = None
        self.mode_coupling_co = None
        self.prop_const = None
        self.core_pitch = None

        # TODO: This may have to get updated in another method since we don't create a new object every time
        self.response = {'SNR': None}

    # TODO: Some dictionary that has all the link information shared between methods, for now, variables
    def _update_link_info(self, link_id):
        link = self.topology_info['links'][link_id]['fiber']

        self.attenuation = link['attenuation']
        self.dispersion = link['dispersion']
        self.bend_radius = link['bending_radius']
        # TODO: Better naming
        self.mode_coupling_co = link['mode_coupling_co']
        self.prop_const = link['propagation_const']
        self.core_pitch = link['core_pitch']

    def _calculate_sci(self, PSDi, BW):
        # TODO: Comment and name better
        Rho = ((math.pi ** 2) * np.abs(self.dispersion)) / (2 * self.attenuation)
        G_SCI = (PSDi ** 2) * math.asinh(Rho * (BW ** 2))
        return G_SCI

    # TODO: I believe repeat code, calculate mci function in routing.py
    def _calculate_link_mci(self, spectrum_contents, slot_index, Fi, MCI):
        # TODO: Better naming or comments (to pylint standards as well)
        # TODO: This line is hard to read, think we're looking for occupied channels?
        # TODO: Rename this variable BW_J and BWj
        BW_J = len(np.where(spectrum_contents == spectrum_contents)[0]) * self.freq_spacing
        Fj = ((slot_index * self.freq_spacing) + ((BW_J) / 2)) * 10 ** 9
        BWj = BW_J * 10 ** 9
        PSDj = self.input_power / BWj
        if Fi != Fj:
            MCI += ((PSDj ** 2) * math.log(abs((abs(Fi - Fj) + (BWj / 2)) / (abs(Fi - Fj) - (BWj / 2)))))

        return MCI

    # TODO: I believe I've implemented something similar in routing (calculate MCI on my branch)
    # TODO: That doesn't have multi-core support I believe, and, probably better to move here instead
    def _calculate_xci(self, Fi, link):
        # TODO: Visited channels or visited spectral slots?
        visited_channels = []
        # TODO: Better naming or constants
        MCI = 0
        for slot_index in range(self.spectral_slots):
            curr_link = self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix']
            spectrum_contents = curr_link[self.spectrum['core_num']][slot_index]

            # Spectrum is occupied
            if spectrum_contents > 0 and spectrum_contents not in visited_channels:
                visited_channels.append(spectrum_contents)
                MCI += self._calculate_link_mci(spectrum_contents=spectrum_contents, slot_index=slot_index, Fi=Fi,
                                                MCI=MCI)

        # TODO: MCI is different on the fifth iteration
        return MCI, visited_channels

    # TODO: Better naming and comments
    def _calculate_pxt(self, length, no_adjacent_core=6):
        xt_eta = (2 * self.bend_radius * self.mode_coupling_co ** 2) / (self.prop_const * self.core_pitch)
        p_xt = no_adjacent_core * xt_eta * length * 1e3 * self.input_power

        return p_xt

    # TODO: Better naming and comments
    def _calculate_xt(self, length, no_adjacent_core=6):
        xt_eta = (2 * self.bend_radius * (self.mode_coupling_co ** 2)) / (self.prop_const * self.core_pitch)
        xt_calc = (1 - math.exp(-2 * xt_eta * length * 1e3)) / (1 + math.exp(-2 * xt_eta * length * 1e3))

        return xt_calc * no_adjacent_core

    # TODO: Better naming and comments
    def _handle_egn(self, visited_channel, length, PSDi, BW, link_id):
        hn = 0
        for i in range(1, math.ceil((len(visited_channel) - 1) / 2) + 1):
            hn = hn + 1 / i
        effective_L = (1 - math.e ** (-2 * self.attenuation * length * 10 ** 3)) / (2 * self.attenuation)

        baud_rate = int(self.req_bit_rate) * 10 ** 9 / 2  # self.baud_rates[self.modulation_format]
        temp_coef = ((self.topology_info['links'][link_id]['fiber']['non_linearity'] ** 2) * (
                effective_L ** 2) * (PSDi ** 3) * (BW ** 2)) / (
                            (baud_rate ** 2) * math.pi * self.dispersion * (length * 10 ** 3))
        PSD_corr = (80 / 81) * self.phi[self.path_mod] * temp_coef * hn

        return PSD_corr

    # TODO: Maybe a more descriptive name of what this does
    #   - Break it up into smaller methods
    #   - Convert this one into a "run" method or something
    # TODO: Better naming and comments
    def SNR_check_NLI_ASE_XT(self):
        # TODO: Move to a constants file or constructor
        light_frequncy = (1.9341 * 10 ** 14)

        # TODO: Probably move to another method
        Fi = ((self.spectrum['start_slot'] * self.freq_spacing) + (
                (self.assigned_slots * self.freq_spacing) / 2)) * 10 ** 9
        BW = self.assigned_slots * self.freq_spacing * 10 ** 9
        PSDi = self.input_power / BW
        # TODO: Define things like this in the constructor
        PSD_NLI = 0
        PSD_corr = 0
        SNR = 0
        for link in range(0, len(self.path) - 1):
            # TODO: Unused variables
            MCI = 0
            # TODO: Is num span used?
            num_span = 0
            # visited_channel = []
            link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']

            self._update_link_info(link_id=link_id)

            # TODO: Probably move to another method
            Mio = (3 * (self.topology_info['links'][link_id]['fiber']['non_linearity'] ** 2)) / (
                    2 * math.pi * self.attenuation * np.abs(self.dispersion))
            G_SCI = self._calculate_sci(PSDi=PSDi, BW=BW)
            G_XCI, visited_channel = self._calculate_xci(Fi=Fi, link=link)
            length = self.topology_info['links'][link_id]['span_length']
            nsp = 1.8  # TODO self.topology_info['links'][link_id]['fiber']['nsp']
            num_span = self.topology_info['links'][link_id]['length'] / length

            # TODO: Probably move to another method
            # TODO: I don't think this variable is needed here
            PSD_ASE = 0
            if self.egn:
                PSD_corr = self._handle_egn(visited_channel=visited_channel, length=length, PSDi=PSDi, BW=BW,
                                            link_id=link_id)
                PSD_NLI = (((G_SCI + G_XCI) * Mio * PSDi)) - PSD_corr
            else:
                PSD_NLI = (((G_SCI + G_XCI) * Mio * PSDi))

            PSD_ASE = (self.plank * light_frequncy * nsp) * (math.exp(self.attenuation * length * 10 ** 3) - 1)
            P_XT = self._calculate_pxt(length=length)
            SNR += (1 / ((PSDi * BW) / (((PSD_ASE + PSD_NLI) * BW + P_XT) * num_span)))

        # TODO: Hard to read
        # TODO: On the fifth iteration, the SNR result is different
        SNR = 10 * math.log10(1 / SNR)
        if SNR == 9.534121780875939:
            print('Begin debug')
        return True if SNR > self.req_snr else False

    def XT_check(self):
        xt = 0
        for link in range(0, len(self.path) - 1):
            link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']
            length = self.topology_info['links'][link_id]['span_length']
            Num_span = self.topology_info['links'][link_id]['length'] / length
            xt += self._calculate_xt(link_id, length) * Num_span
        xt = 10 * math.log10(xt)
        # TODO: Hard to read
        return True if xt < self.requested_xt else False
