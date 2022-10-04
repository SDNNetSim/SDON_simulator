import unittest
import numpy as np

from scripts.routing import Routing
from scripts.engine import Engine


class TestRouting(unittest.TestCase):
    """
    Tests the routing methods.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        self.engine = Engine(sim_input_fp='../tests/data/input2.json')
        self.engine.load_input()
        self.engine.create_pt()

        self.physical_topology = self.engine.physical_topology
        self.network_spec_db = self.engine.network_spec_db

        self.routing = Routing(source='Lowell', destination='San Francisco', physical_topology=self.physical_topology,
                               network_spec_db=self.network_spec_db)

    def test_least_cong_route(self):
        test_dict0 = {'path': ['Lowell', 'Las Vegas', 'Austin'], 'link_info': {'slots_taken': 10}}
        test_dict1 = {'path': ['Chicago', 'Houston', 'Richmond', 'Los Angeles'], 'link_info': {'slots_taken': 2}}
        test_dict2 = {'path': ['Boston', 'Miami'], 'link_info': {'slots_taken': 15}}
        test_dict3 = {'path': ['Boston', 'Lowell', 'Nashua'], 'link_info': {'slots_taken': 300}}
        self.routing.paths_list = [test_dict0, test_dict1, test_dict2, test_dict3]

        check_list = self.routing.find_least_cong_route()

        self.assertEqual(['Chicago', 'Houston', 'Richmond', 'Los Angeles'], check_list, 'Incorrect path chosen.')

    def test_find_most_cong_link(self):
        # For the first path let's do link number 2
        # For the second path let's do link 6
        path1 = ['Lowell', 'Boston', 'Miami', 'Chicago', 'San Francisco']
        path2 = ['Lowell', 'Las Vegas', 'San Francisco']
        # Congest link 2
        self.network_spec_db[('Boston', 'Miami')]['cores_matrix'][0][10:20] = 1
        # Congest link 6
        self.network_spec_db[('Las Vegas', 'San Francisco')]['cores_matrix'][0][0:] = 1
        self.network_spec_db[('Las Vegas', 'San Francisco')]['cores_matrix'][1][0:] = 1
        self.network_spec_db[('Las Vegas', 'San Francisco')]['cores_matrix'][2][0:] = 1

        self.routing.find_most_cong_link(path=path1)
        self.routing.find_most_cong_link(path=path2)

        self.assertEqual({'path': path1, 'link_info': {'link': '2', 'slots_taken': 10}}, self.routing.paths_list[0])
        self.assertEqual({'path': path2, 'link_info': {'link': '6', 'slots_taken': 1500}}, self.routing.paths_list[1])

    def test_least_congested_path(self):
        pass
