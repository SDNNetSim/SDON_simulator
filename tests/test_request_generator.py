import unittest
import os
import json

from sim_scripts.request_generator import generate


class TestRequestGenerator(unittest.TestCase):
    """
    Tests the request generator function.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        working_dir = os.getcwd().split('/')
        if working_dir[-1] == 'tests':
            file_path = '../tests/test_data/'
        else:
            file_path = './tests/test_data/'

        with open(file_path + 'bandwidth_info.json', 'r', encoding='utf-8') as file_obj:
            self.mod_formats = json.load(file_obj)

        with open(file_path + 'input3.json', 'r', encoding='utf-8') as file_obj:
            self.sim_input = json.load(file_obj)

    def test_request_ratios(self):
        """
        Test to ensure we have the correct ratio of bandwidths.
        """
        resp = generate(seed_no=1,
                        nodes=list(self.sim_input['physical_topology']['nodes'].keys()),
                        mu=self.sim_input['mu'],
                        lam=self.sim_input['lambda'],
                        num_requests=1000,
                        bw_dict=self.mod_formats,
                        assume='yue')

        bandwidth_ratios = {'50': 0, '100': 0, '400': 0}
        for time, obj in resp.items():  # pylint: disable=unused-variable
            bandwidth_ratios[obj['bandwidth']] += 1

        # Numbers are doubled due to arrival and departure requests being counted
        check_obj = {'50': 600, '100': 1000, '400': 400}
        self.assertEqual(check_obj, bandwidth_ratios,
                         f'Incorrect request bandwidth ratio generated. Expected {check_obj} and got '
                         f'{bandwidth_ratios}')
