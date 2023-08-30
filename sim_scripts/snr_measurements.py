import math

import numpy as np


class SnrMeasurements:
    """
    Calculates SNR for a given request.
    """

    # TODO: Might be a good idea to reference all constants to a paper or something
    # TODO: Move variables as needed to the configuration file
    # TODO: Change calculate methods to get
    def __init__(self, path=None, path_mod=None, spectrum=None, assigned_slots=None, req_bit_rate=12.5,
                 freq_spacing=12.5, input_power=10 ** -3, spectral_slots=None, req_snr=8.5, net_spec_db=None,
                 topology_info=None, guard_slots=0, baud_rates=None, egn_model=False, xt_noise=False,
                 bi_directional=True, requested_xt=-30):

        # TODO: If these values will not change, it's best to take them out of the params of the constructor
        # TODO: Comments for all of these or move to a params dict
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
        # Flag to show a use of the EGN model or the GN model for SNR calculations
        self.egn_model = egn_model
        # A parameter related to the EGN model
        self.phi = {'QPSK': 1, '16-QAM': 0.68, '64-QAM': 0.6190476190476191}
        self.requests_status = {}
        self.baud_rates = baud_rates
        self.bi_directional = bi_directional
        self.xt_noise = xt_noise
        self.plank = 6.62607004e-34
        self.requested_xt = requested_xt

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
        self.light_frequency = 1.9341 * 10 ** 14

    # TODO: Some dictionary that has all the link information shared between methods, for now, variables
    def _update_link_info(self):
        self.link = self.topology_info['links'][self.link_id]['fiber']
        self.attenuation = self.link['attenuation']
        self.dispersion = self.link['dispersion']
        self.bend_radius = self.link['bending_radius']
        self.coupling_coeff = self.link['mode_coupling_co']
        self.prop_const = self.link['propagation_const']
        self.core_pitch = self.link['core_pitch']

    def _init_center_freq(self):
        self.center_freq = ((self.spectrum['start_slot'] * self.freq_spacing) + (
                (self.assigned_slots * self.freq_spacing) / 2)) * 10 ** 9
        self.bandwidth = self.assigned_slots * self.freq_spacing * 10 ** 9
        self.center_psd = self.input_power / self.bandwidth

    def _calculate_sci_psd(self):
        rho_param = ((math.pi ** 2) * np.abs(self.dispersion)) / (2 * self.attenuation)
        sci_psd = (self.center_psd ** 2) * math.asinh(rho_param * (self.bandwidth ** 2))
        return sci_psd

    # TODO: I believe repeat code, calculate mci function in routing.py
    def _update_link_mci(self, spectrum_contents, curr_link, slot_index, curr_mci):
        num_slots = len(np.where(spectrum_contents == curr_link[self.spectrum['core_num']])[0]) * self.freq_spacing
        channel_freq = ((slot_index * self.freq_spacing) + (num_slots / 2)) * 10 ** 9

        channel_bw = num_slots * 10 ** 9
        channel_psd = self.input_power / channel_bw
        if self.center_freq != channel_freq:
            new_mci = curr_mci + ((channel_psd ** 2) * math.log(
                abs((abs(self.center_freq - channel_freq) + (channel_bw / 2)) / (
                        abs(self.center_freq - channel_freq) - (channel_bw / 2)))))
        else:
            new_mci = curr_mci

        return new_mci

    # TODO: I believe I've implemented something similar in routing (calculate MCI on my branch)
    # TODO: That doesn't have multi-core support I believe, and, probably better to move here instead
    def _calculate_xci(self, link):
        self.visited_channels = []
        # Cross-phase modulation noise
        xci_noise = 0
        for slot_index in range(self.spectral_slots):
            curr_link = self.net_spec_db[(self.path[link], self.path[link + 1])]['cores_matrix']
            spectrum_contents = curr_link[self.spectrum['core_num']][slot_index]

            # Spectrum is occupied
            if spectrum_contents > 0 and spectrum_contents not in self.visited_channels:
                self.visited_channels.append(spectrum_contents)
                xci_noise = self._update_link_mci(spectrum_contents=spectrum_contents, curr_link=curr_link,
                                                  slot_index=slot_index, curr_mci=xci_noise)

        return xci_noise

    def _calculate_pxt(self, adjacent_cores=6):
        # A statistical mean of the cross-talk
        mean_xt = (2 * self.bend_radius * self.coupling_coeff ** 2) / (self.prop_const * self.core_pitch)
        # The cross-talk noise power
        power_xt = adjacent_cores * mean_xt * self.length * 1e3 * self.input_power

        return power_xt

    def _calculate_xt(self, adjacent_cores=6):
        mean_xt = (2 * self.bend_radius * (self.coupling_coeff ** 2)) / (self.prop_const * self.core_pitch)
        # TODO: Define this in the docstring and name it 'resp'
        resp_xt = (1 - math.exp(-2 * mean_xt * self.length * 1e3)) / (1 + math.exp(-2 * mean_xt * self.length * 1e3))

        return resp_xt * adjacent_cores

    def _handle_egn_model(self):
        # The harmonic number series
        hn_series = 0
        for i in range(1, math.ceil((len(self.visited_channels) - 1) / 2) + 1):
            hn_series = hn_series + 1 / i

        # The effective span length
        eff_span_len = (1 - math.e ** (-2 * self.attenuation * self.length * 10 ** 3)) / (2 * self.attenuation)

        baud_rate = int(self.req_bit_rate) * 10 ** 9 / 2
        # TODO: Break up these equations to other variables
        temp_coef = ((self.topology_info['links'][self.link_id]['fiber']['non_linearity'] ** 2) * (
                eff_span_len ** 2) * (self.center_psd ** 3) * (self.bandwidth ** 2)) / (
                            (baud_rate ** 2) * math.pi * self.dispersion * (self.length * 10 ** 3))
        # The PSD correction term
        psd_correction = (80 / 81) * self.phi[self.path_mod] * temp_coef * hn_series

        return psd_correction

    def _update_link_params(self, link):
        self.mu_param = (3 * (self.topology_info['links'][self.link_id]['fiber']['non_linearity'] ** 2)) / (
                2 * math.pi * self.attenuation * np.abs(self.dispersion))
        self.sci_psd = self._calculate_sci_psd()
        self.xci_psd = self._calculate_xci(link=link)
        self.length = self.topology_info['links'][self.link_id]['span_length']
        # TODO Add support for self.topology_info['links'][link_id]['fiber']['nsp']
        self.nsp = 1.8
        self.num_span = self.topology_info['links'][self.link_id]['length'] / self.length

    def _calculate_psd_nli(self):
        if self.egn_model:
            psd_correction = self._handle_egn_model()
            psd_nli = ((self.sci_psd + self.xci_psd) * self.mu_param * self.center_psd) - psd_correction
        else:
            psd_nli = ((self.sci_psd + self.xci_psd) * self.mu_param * self.center_psd)

        return psd_nli

    def check_snr(self):
        snr = 0
        self._init_center_freq()
        for link in range(0, len(self.path) - 1):
            self.link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']

            self._update_link_info()
            self._update_link_params(link=link)

            psd_nli = self._calculate_psd_nli()

            psd_ase = (self.plank * self.light_frequency * self.nsp) * (
                    math.exp(self.attenuation * self.length * 10 ** 3) - 1)
            mean_xt = self._calculate_pxt()
            snr += (1 / ((self.center_psd * self.bandwidth) / (
                    ((psd_ase + psd_nli) * self.bandwidth + mean_xt) * self.num_span)))

        snr = 10 * math.log10(1 / snr)

        resp = snr > self.req_snr
        return resp

    # TODO: I don't believe this is used
    def check_xt(self):
        cross_talk = 0

        for link in range(0, len(self.path) - 1):
            self.link_id = self.net_spec_db[(self.path[link], self.path[link + 1])]['link_num']
            self.length = self.topology_info['links'][self.link_id]['span_length']
            self.num_span = self.topology_info['links'][self.link_id]['length'] / self.length
            cross_talk += self._calculate_xt() * self.num_span

        cross_talk = 10 * math.log10(cross_talk)
        resp = cross_talk < self.requested_xt
        return resp
