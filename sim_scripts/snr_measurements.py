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

        self.Fi = None
        self.BW = None
        self.PSDi = None

        self.Mio = None
        self.G_SCI = None
        self.G_XCI = None
        self.visited_channel = None
        self.length = None
        self.nsp = None
        self.num_span = None
        self.link_id = None

        # TODO: This may have to get updated in another method since we don't create a new object every time
        self.response = {'SNR': None}
        # TODO: Move to a constants file or constructor
        self.light_frequency = (1.9341 * 10 ** 14)

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

    def _init_vars(self):
        # TODO: Better naming
        self.Fi = ((self.spectrum['start_slot'] * self.freq_spacing) + (
                (self.assigned_slots * self.freq_spacing) / 2)) * 10 ** 9
        self.BW = self.assigned_slots * self.freq_spacing * 10 ** 9
        self.PSDi = self.input_power / self.BW

    def _calculate_sci(self, PSDi, BW):
        # TODO: Comment and name better
        Rho = ((math.pi ** 2) * np.abs(self.dispersion)) / (2 * self.attenuation)
        G_SCI = (PSDi ** 2) * math.asinh(Rho * (BW ** 2))
        return G_SCI

    # TODO: I believe repeat code, calculate mci function in routing.py
    # TODO: Move link to constructor if used in multiple methods
    def _calculate_link_mci(self, spectrum_contents, curr_link, slot_index, Fi, MCI):
        # TODO: Better naming or comments (to pylint standards as well)
        # TODO: This line is hard to read, think we're looking for occupied channels?
        # TODO: Rename this variable BW_J and BWj (BW_J is different)
        # TODO: Set equal to the core number instead of "itself", spectrum contents has to change
        BW_J = len(np.where(spectrum_contents == curr_link[self.spectrum['core_num']])[0]) * self.freq_spacing
        # TODO: Here is the issue (slot index)
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
                # TODO: Slot indexes may be different
                # TODO: MCI is different
                MCI = self._calculate_link_mci(spectrum_contents=spectrum_contents, curr_link=curr_link,
                                               slot_index=slot_index, Fi=Fi, MCI=MCI)

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

    # TODO: Probably an incorrect name
    def _calculate_mci(self, link_id, link):
        # TODO: Probably move to another method
        self.Mio = (3 * (self.topology_info['links'][link_id]['fiber']['non_linearity'] ** 2)) / (
                2 * math.pi * self.attenuation * np.abs(self.dispersion))
        self.G_SCI = self._calculate_sci(PSDi=self.PSDi, BW=self.BW)
        self.G_XCI, self.visited_channel = self._calculate_xci(Fi=self.Fi, link=link)
        self.length = self.topology_info['links'][link_id]['span_length']
        self.nsp = 1.8  # TODO self.topology_info['links'][link_id]['fiber']['nsp']
        self.num_span = self.topology_info['links'][link_id]['length'] / self.length

    def _calculate_psd_nli(self):
        if self.egn:
            PSD_corr = self._handle_egn(visited_channel=self.visited_channel, length=self.length, PSDi=self.PSDi,
                                        BW=self.BW, link_id=self.link_id)
            PSD_NLI = (((self.G_SCI + self.G_XCI) * self.Mio * self.PSDi)) - PSD_corr
        else:
            PSD_NLI = (((self.G_SCI + self.G_XCI) * self.Mio * self.PSDi))

        return PSD_NLI

    # TODO: Maybe a more descriptive name of what this does
    #   - Break it up into smaller methods
    #   - Convert this one into a "run" method or something
    # TODO: Better naming and comments
    def SNR_check_NLI_ASE_XT(self):
        self._init_vars()
        # TODO: Define things like this in the constructor
        SNR = 0
        for link in range(0, len(self.path) - 1):
            self.link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']

            # TODO: Define in constructor (used in multiple methods)
            self._update_link_info(link_id=self.link_id)
            self._calculate_mci(link_id=self.link_id, link=link)

            PSD_NLI = self._calculate_psd_nli()

            PSD_ASE = (self.plank * self.light_frequency * self.nsp) * (
                        math.exp(self.attenuation * self.length * 10 ** 3) - 1)
            P_XT = self._calculate_pxt(length=self.length)
            SNR += (1 / ((self.PSDi * self.BW) / (((PSD_ASE + PSD_NLI) * self.BW + P_XT) * self.num_span)))

        # TODO: Hard to read
        # TODO: On the fifth iteration, the SNR result is different
        SNR = 10 * math.log10(1 / SNR)
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
