# pylint: disable=protected-access

import unittest
import math

import networkx as nx
import numpy as np

from fixtures.test_snr_args import engine_props, sdn_props, snr_props
from sim_scripts.snr_measurements import SnrMeasurements


class TestSnrMeasurements(unittest.TestCase):
    """
    Tests snr_measurements.py
    """

    def setUp(self):
        self.engine_props = engine_props
        self.topology = nx.Graph()
        self.topology.add_edge('source', 'dest', length=100)
        self.topology.add_edge('source', 'intermediate', length=80)
        self.topology.add_edge('intermediate', 'dest', length=120)
        self.engine_props['topology'] = self.topology

        self.sdn_props = sdn_props
        self.spectrum_props = {
            'core_num': 0,
            'path_list': ['source', 'dest'],
            'modulation': 'QPSK',
            'start_slot': 2,
            'end_slot': 3,
        }
        self.snr_props = snr_props
        self.snr_measurements = SnrMeasurements(self.engine_props, self.sdn_props, self.spectrum_props)
        self.snr_measurements.snr_props = self.snr_props

        self.link_id = 'link_id'
        self.snr_measurements.link_id = self.link_id
        self.channels_list = [1, 2, 3]
        self.snr_measurements.channels_list = self.channels_list
        self.snr_measurements.num_slots = 2

    def test_calculate_sci_psd(self):
        """
        Test calculate SCI-PSD.
        """
        snr_measurements = SnrMeasurements(self.engine_props, self.sdn_props, self.spectrum_props)
        snr_measurements.snr_props = self.snr_props

        rho_param = (math.pi ** 2) * np.abs(self.snr_props['link_dict']['dispersion'])
        rho_param /= (2 * self.snr_props['link_dict']['attenuation'])
        expected_sci_psd = self.snr_props['center_psd'] ** 2
        expected_sci_psd *= math.asinh(rho_param * (self.snr_props['bandwidth'] ** 2))

        result_sci_psd = snr_measurements._calculate_sci_psd()
        self.assertAlmostEqual(result_sci_psd, expected_sci_psd, places=5,
                               msg="SCI PSD calculation did not match expected value.")

    def test_update_link_xci(self):
        """
        Test update link XCI.
        """
        req_id = 1.0
        curr_link = np.array([req_id, 0, 0, req_id])
        slot_index = 1
        curr_xci = 0.0

        channel_bw = 2 * self.engine_props['bw_per_slot']
        channel_freq = ((slot_index * self.engine_props['bw_per_slot']) + (channel_bw / 2)) * 10 ** 9
        channel_bw *= 10 ** 9
        channel_psd = self.engine_props['input_power'] / channel_bw

        self.snr_props['center_freq'] = 193.1e12
        if self.snr_props['center_freq'] != channel_freq:
            log_term = abs(self.snr_props['center_freq'] - channel_freq) + (channel_bw / 2)
            log_term /= abs(self.snr_props['center_freq'] - channel_freq) - (channel_bw / 2)
            calculated_xci = (channel_psd ** 2) * math.log(abs(log_term))
            expected_new_xci = curr_xci + calculated_xci
        else:
            expected_new_xci = curr_xci

        result_xci = self.snr_measurements._update_link_xci(req_id, curr_link, slot_index, curr_xci)
        self.assertAlmostEqual(result_xci, expected_new_xci, places=5,
                               msg="Updated XCI does not match expected value.")

    def test_calculate_xci(self):
        """
        Test calculate XCI.
        """
        link_num = 0
        result_xci = self.snr_measurements._calculate_xci(link_num)
        self.assertIsInstance(result_xci, float, msg="XCI calculation did not return a float value.")

    def test_calculate_pxt(self):
        """
        Test calculate PXT.
        """
        num_adjacent = 2
        mean_xt = 2 * self.snr_props['link_dict']['bending_radius']
        mean_xt *= self.snr_props['link_dict']['mode_coupling_co'] ** 2
        mean_xt /= (self.snr_props['link_dict']['propagation_const'] * self.snr_props['link_dict']['core_pitch'])
        expected_power_xt = num_adjacent * mean_xt * self.snr_props['length'] * 1e3 * self.engine_props['input_power']

        result_power_xt = self.snr_measurements._calculate_pxt(num_adjacent)
        self.assertAlmostEqual(result_power_xt, expected_power_xt, places=5,
                               msg="Calculated cross-talk noise power does not match expected value.")

    def test_calculate_xt(self):
        """
        Test calculate XT.
        """
        num_adjacent = 2
        link_length = 100
        mean_xt = 3.78e-9
        resp_xt = 1 - math.exp(-2 * mean_xt * link_length * 1e3)
        resp_xt /= (1 + math.exp(-2 * mean_xt * link_length * 1e3))
        expected_resp_xt = resp_xt * num_adjacent

        result_resp_xt = SnrMeasurements.calculate_xt(num_adjacent, link_length)
        self.assertAlmostEqual(result_resp_xt, expected_resp_xt, places=5,
                               msg="Calculated cross-talk interference does not match expected value.")

    def test_handle_egn_model(self):
        """
        Test handle EGN model.
        """
        self.snr_measurements.channels_list = [1, 2, 3]
        hn_series = sum(1 / i for i in range(1, math.ceil((len(self.snr_measurements.channels_list) - 1) / 2) + 1))
        power = -2 * self.snr_props['link_dict']['attenuation'] * self.snr_props['length'] * 1e3
        eff_span_len = (1 - math.exp(power)) / (2 * self.snr_props['link_dict']['attenuation'])
        baud_rate = self.snr_props['req_bit_rate'] * 1e9 / 2  # Convert Gbps to bps and divide by 2

        temp_coef = self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2
        temp_coef *= eff_span_len ** 2
        temp_coef *= (self.snr_props['center_psd'] ** 3 * self.snr_props['bandwidth'] ** 2)
        temp_coef /= (baud_rate ** 2 * math.pi * self.snr_props['link_dict']['dispersion'] * (
                self.snr_props['length'] * 1e3))

        expected_psd_correction = (80 / 81) * self.engine_props['phi'][self.spectrum_props['modulation']]
        expected_psd_correction *= temp_coef * hn_series
        psd_correction = self.snr_measurements._handle_egn_model()

        self.assertAlmostEqual(psd_correction, expected_psd_correction, places=5,
                               msg="Calculated PSD correction does not match expected value.")

    def test_calculate_psd_nli(self):
        """
        Test calculate PSD NLI.
        """
        expected_psd_nli = self.snr_props['sci_psd'] + self.snr_props['xci_psd']
        expected_psd_nli *= (self.snr_props['mu_param'] * self.snr_props['center_psd'])

        if self.engine_props['egn_model']:
            psd_correction = self.snr_measurements._handle_egn_model()
            expected_psd_nli -= psd_correction

        psd_nli = self.snr_measurements._calculate_psd_nli()
        self.assertAlmostEqual(psd_nli, expected_psd_nli, places=5,
                               msg="Calculated PSD NLI does not match expected value.")

    def test_update_link_params(self):
        """
        Test update link params.
        """
        link_num = 0
        self.snr_measurements._update_link_params(link_num=link_num)

        non_linearity = self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2
        expected_mu_param = 3 * non_linearity
        mu_denominator = 2 * math.pi * self.snr_props['link_dict']['attenuation'] * np.abs(
            self.snr_props['link_dict']['dispersion'])
        expected_mu_param /= mu_denominator

        expected_length = self.engine_props['topology_info']['links'][self.link_id]['span_length']
        link_length = self.engine_props['topology_info']['links'][self.link_id]['length']
        expected_num_span = link_length / expected_length

        self.assertAlmostEqual(self.snr_props['mu_param'], expected_mu_param, places=5,
                               msg="mu_param not updated correctly.")
        self.assertIsInstance(self.snr_props['sci_psd'], float, msg="sci_psd not updated correctly or not a float.")
        self.assertIsInstance(self.snr_props['xci_psd'], float, msg="xci_psd not updated correctly or not a float.")
        self.assertEqual(self.snr_props['length'], expected_length, msg="Link length not updated correctly.")
        self.assertEqual(self.snr_props['num_span'], expected_num_span, msg="Number of spans not updated correctly.")

    def test_check_snr(self):
        """
        Test check SNR.
        """
        psd_ase = self.snr_props['plank'] * self.snr_props['light_frequency'] * self.snr_props['nsp']
        psd_ase *= (math.exp(self.snr_props['link_dict']['attenuation'] * self.snr_props['length'] * 1e3) - 1)
        psd_nli = (self.snr_props['sci_psd'] + self.snr_props['xci_psd']) * (
                self.snr_props['mu_param'] * self.snr_props['center_psd'])
        p_xt = 0

        curr_snr = self.snr_props['center_psd'] * self.snr_props['bandwidth']
        curr_snr /= (((psd_ase + psd_nli) * self.snr_props['bandwidth'] + p_xt) * self.snr_props['num_span'])
        total_snr = 10 * math.log10(curr_snr)  # Convert to dB
        expected_snr_met = total_snr > self.snr_props['req_snr']

        snr_met = self.snr_measurements.check_snr()
        self.assertEqual(snr_met, expected_snr_met, msg="SNR check did not return the expected result.")

    def test_check_adjacent_cores(self):
        """
        Test check adjacent cores.
        """
        link_tuple = ('source', 'dest')
        num_adjacent_cores = self.snr_measurements.check_adjacent_cores(link_tuple)
        expected_num_adjacent_cores = 2

        self.assertEqual(num_adjacent_cores, expected_num_adjacent_cores,
                         msg="Number of adjacent cores with overlapping channels does not match expected value.")

    def test_find_worst_xt(self):
        """
        Test find the worst XT.
        """
        flag = 'intra_core'
        resp, max_length = self.snr_measurements.find_worst_xt(flag)

        expected_max_length = 120
        expected_resp = self.snr_measurements.calculate_xt(num_adjacent=6, link_length=expected_max_length)
        expected_resp = 10 * math.log10(expected_resp)

        self.assertAlmostEqual(max_length, expected_max_length, msg="Max link length does not match expected value.")
        self.assertAlmostEqual(resp, expected_resp, msg="Calculated XT does not match expected value.")

    def test_check_xt(self):
        """
        Test check XT.
        """
        num_adjacent = 2
        xt_per_link = []
        for link_num in range(len(self.spectrum_props['path_list']) - 1):
            link_tuple = (self.spectrum_props['path_list'][link_num], self.spectrum_props['path_list'][link_num + 1])
            self.link_id = self.sdn_props['net_spec_dict'][link_tuple]['link_num']
            link_length = self.engine_props['topology_info']['links'][self.link_id]['length']
            curr_xt = self.snr_measurements.calculate_xt(num_adjacent=num_adjacent, link_length=link_length)
            xt_per_link.append(curr_xt)

        total_xt = sum(xt_per_link)
        expected_cross_talk = 10 * math.log10(total_xt) if total_xt > 0 else 0
        expected_xt_met = expected_cross_talk < self.engine_props['requested_xt'][self.spectrum_props['modulation']]

        xt_met, cross_talk = self.snr_measurements.check_xt()
        self.assertEqual(xt_met, expected_xt_met, "XT check did not return the expected result.")
        self.assertAlmostEqual(cross_talk, expected_cross_talk, places=5,
                               msg="Calculated cross-talk does not match expected value.")
