import unittest
import os
import json

from arg_scripts.data_args import YUE_MOD_ASSUMPTIONS, ARASH_MOD_ASSUMPTIONS
from data_scripts.generate_data import create_pt, create_bw_info


class TestGenerateData(unittest.TestCase):
    """
    Tests functions in generate_data.py script.
    """

    def test_create_pt(self):
        """
        Tests creating physical topology.
        """
        cores_per_link = 4
        net_spec_dict = {('A', 'B'): 100, ('B', 'C'): 150}

        topology_path = os.path.join('.', 'fixtures', 'topology.json')
        with open(topology_path, 'r', encoding='utf-8') as file_obj:
            exp_top_dict = json.load(file_obj)

            # When loading a json, the keys are strings and no longer integers...
            # This means I should change the data structure, but I'm not going to
            check_top_dict = {'nodes': exp_top_dict['nodes'], 'links': {}}
            for link_num, data_dict in exp_top_dict['links'].items():
                check_top_dict['links'][int(link_num)] = data_dict

        topology_dict = create_pt(cores_per_link, net_spec_dict)
        self.assertEqual(topology_dict, check_top_dict)

    def test_create_bw_info_yue(self):
        """
        Tests creating Yue's bandwidth assumptions.
        """
        sim_type = 'yue'
        expected_bw_mod_dict = YUE_MOD_ASSUMPTIONS
        bw_mod_dict = create_bw_info(sim_type)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict)

    def test_create_bw_info_arash(self):
        """
        Tests creating Arash's bandwidth assumptions.
        """
        sim_type = 'arash'
        expected_bw_mod_dict = ARASH_MOD_ASSUMPTIONS
        bw_mod_dict = create_bw_info(sim_type)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict)

    def test_create_bw_info_invalid(self):
        """
        Tests and invalid bandwidth assumption.
        """
        sim_type = 'invalid'
        with self.assertRaises(NotImplementedError):
            create_bw_info(sim_type)
