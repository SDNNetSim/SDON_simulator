import unittest
import os
import json

import numpy as np

from sim_scripts.engine import Engine
from sim_scripts.sdn_controller import SDNController


class TestSDNController(unittest.TestCase):
    """
    Tests the SDN controller class.
    """

    def setUp(self):
        """
        Set up the class for testing.
        """
        # TODO: Eventually have one setup script for creating topology and things like that
        working_dir = os.getcwd().split('/')
        if working_dir[-1] == 'tests':
            file_path = './test_data/'
        else:
            file_path = './tests/test_data/'

        self.engine = Engine(sim_input_fp=file_path + 'input3.json')
        self.engine.load_input()
        self.engine.create_pt()

        self.topology = self.engine.physical_topology
        self.network_db = self.engine.network_spec_db

        with open(file_path + 'bandwidth_info.json', "r") as file_obj:
            mod_formats = json.load(file_obj)

        self.controller = SDNController(req_id=7, network_db=self.network_db, topology=self.topology, num_cores=1,
                                        mod_formats=mod_formats, sim_assume='arash')
        self.controller.path = ['Lowell', 'Miami', 'Chicago', 'San Francisco']
        self.controller.guard_band = 1
        self.controller.req_id = 7
        self.controller.chosen_bw = '100'
        self.controller.max_lps = 4

    def test_handle_release(self):
        """
        Test that a link has been released of its resources properly.
        """
        for i in range(len(self.controller.path) - 1):
            src_dest = (self.controller.path[i], self.controller.path[i + 1])
            dest_src = (self.controller.path[i + 1], self.controller.path[i])

            self.network_db[src_dest]['cores_matrix'][0][5:15] = self.controller.req_id
            # For the guard band
            self.network_db[src_dest]['cores_matrix'][0][15] = -self.controller.req_id

            self.network_db[dest_src]['cores_matrix'][0][5:15] = self.controller.req_id
            self.network_db[dest_src]['cores_matrix'][0][15] = -self.controller.req_id

        self.controller.network_db = self.network_db
        self.controller.handle_release()

        for i in range(len(self.controller.path) - 1):
            # TODO: Assuming dest_src works if src_dest works (remember, we have bi-directional links)
            src_dest = (self.controller.path[i], self.controller.path[i + 1])

            req_indexes = np.where(self.controller.network_db[src_dest]['cores_matrix'][0] == self.controller.req_id)[0]
            gb_indexes = np.where(self.controller.network_db[src_dest]['cores_matrix'][0] == -self.controller.req_id)[0]

            self.assertEqual(0, len(req_indexes), 'The request still exists in the network.')
            self.assertEqual(0, len(gb_indexes), 'The guard band still exists in the network.')

    def test_handle_arrival(self):
        """
        Test that a link has allocated its resources properly.
        """
        self.controller.network_db = self.network_db
        self.controller.handle_arrival(core_num=0, start_slot=20, end_slot=25)

        for i in range(len(self.controller.path) - 1):
            src_dest = (self.controller.path[i], self.controller.path[i + 1])

            req_arr = self.controller.network_db[src_dest]['cores_matrix'][0][20:24]
            gb = self.controller.network_db[src_dest]['cores_matrix'][0][24]

            exp_arr = [7.0, 7.0, 7.0, 7.0]
            self.assertEqual(exp_arr, list(req_arr),
                             f'Request was not allocated properly, expected {exp_arr} and got {req_arr}')
            self.assertEqual(-7, gb, f'Guard band was not allocated properly. Expected -10 and got {gb}')

    def test_single_core_lps(self):
        """
        Test that a request has been sliced into light segments properly in a single core.
        """
        self.controller.network_db = self.network_db
        self.controller.single_core = True

        # Congest all links except the end and beginning of the array
        for i in range(len(self.controller.path) - 1):
            src_dest = (self.controller.path[i], self.controller.path[i + 1])
            dest_src = (self.controller.path[i + 1], self.controller.path[i])

            self.network_db[src_dest]['cores_matrix'][0][2:-2] = 1
            self.network_db[dest_src]['cores_matrix'][0][2:-2] = 1

        self.controller.handle_lps()

        # Ensure no slicing occurred on anything but the first core
        check_arr = self.controller.network_db[('Lowell', 'Miami')]['cores_matrix'][1]
        ind_arr = np.where((check_arr == 7) | (check_arr == -7))[0]
        self.assertEqual(0, len(ind_arr), 'Slicing occurred on multiple cores.')

        # Ensure slicing occurred in the expected indexes
        check_arr = self.controller.network_db[('Lowell', 'Miami')]['cores_matrix'][0]
        ind_arr = np.where((check_arr == 7) | (check_arr == -7))[0]
        exp_ind = [0, 1, 498, 499]
        self.assertTrue(np.alltrue(ind_arr == exp_ind),
                        f'Slicing occurred in indexes {ind_arr} when it should have occurred in {exp_ind}')

    def test_multi_core_lps(self):
        """
        Test that a request has been sliced into light segments properly in multiple cores.
        """
        self.controller.single_core = False

        # Congest all links except the beginning of the array
        for i in range(len(self.controller.path) - 1):
            src_dest = (self.controller.path[i], self.controller.path[i + 1])
            dest_src = (self.controller.path[i + 1], self.controller.path[i])

            self.network_db[src_dest]['cores_matrix'][0][2:] = 1
            self.network_db[dest_src]['cores_matrix'][0][2:] = 1

            self.network_db[src_dest]['cores_matrix'][1][0:-2] = 1
            self.network_db[dest_src]['cores_matrix'][1][0:-2] = 1

        self.controller.handle_lps()

        # Check the first core
        check_arr = self.controller.network_db[('Lowell', 'Miami')]['cores_matrix'][0]
        ind_arr = np.where((check_arr == 7) | (check_arr == -7))[0]
        exp_ind = [0, 1]
        self.assertTrue(np.alltrue(ind_arr == exp_ind),
                        f'Slicing occurred in indexes {ind_arr} when it should have occurred in {exp_ind}')

        # Check the second core
        check_arr = self.controller.network_db[('Lowell', 'Miami')]['cores_matrix'][1]
        ind_arr = np.where((check_arr == 7) | (check_arr == -7))[0]
        exp_ind = [498, 499]
        self.assertTrue(np.alltrue(ind_arr == exp_ind),
                        f'Slicing occurred in indexes {ind_arr} when it should have occurred in {exp_ind}')


if __name__ == '__main__':
    unittest.main()
