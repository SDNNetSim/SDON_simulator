import unittest
import os

from sim_scripts.routing import Routing
from sim_scripts.engine import Engine


class TestRouting(unittest.TestCase):
    """
    Tests the methods in the routing class.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        working_dir = os.getcwd().split('/')
        if working_dir[-1] == 'tests':
            file_path = '../tests/test_data/input3.json'
        else:
            file_path = './tests/test_data/input3.json'

        self.engine = Engine(sim_input_fp=file_path)
        self.engine.load_input()
        self.engine.create_pt()

        self.physical_topology = self.engine.physical_topology
        self.network_spec_db = self.engine.network_spec_db

        mod_formats = {'QPSK': {'max_length': 50000}, '16-QAM': {'max_length': 1}}

        self.routing = Routing(source='Lowell', destination='San Francisco', physical_topology=self.physical_topology,
                               network_spec_db=self.network_spec_db, mod_formats=mod_formats)

    def test_least_cong_route(self):
        """
        Test the least congested route method.
        """
        test_dict0 = {'path': ['Lowell', 'Las Vegas', 'Austin'], 'link_info': {'free_slots': 0}}
        test_dict1 = {'path': ['Chicago', 'Houston', 'Richmond', 'Los Angeles'], 'link_info': {'free_slots': 32}}
        test_dict2 = {'path': ['Boston', 'Miami'], 'link_info': {'free_slots': 127}}
        test_dict3 = {'path': ['Boston', 'Lowell', 'Nashua'], 'link_info': {'free_slots': 128}}
        self.routing.paths_list = [test_dict0, test_dict1, test_dict2, test_dict3]

        check_list = self.routing.find_least_cong_route()

        self.assertEqual(['Boston', 'Lowell', 'Nashua'], check_list, 'Incorrect path chosen.')

    def test_find_most_cong_link(self):
        """
        Tests the find most congested link method by forcing the simulation to choose a specific link.
        """
        path1 = ['Lowell', 'Miami', 'Chicago', 'San Francisco']
        path2 = ['Lowell', 'Las Vegas', 'Portland', 'San Francisco']
        # Congest link 1
        self.network_spec_db[('Lowell', 'Miami')]['cores_matrix'][0][10:20] = 1
        # Congest link 5
        self.network_spec_db[('Las Vegas', 'Portland')]['cores_matrix'][0][0:] = 1
        self.network_spec_db[('Las Vegas', 'Portland')]['cores_matrix'][1][0:] = 1
        self.network_spec_db[('Las Vegas', 'Portland')]['cores_matrix'][2][0:] = 1

        self.routing.find_most_cong_link(path=path1)
        self.routing.find_most_cong_link(path=path2)

        self.assertEqual({'path': path1, 'link_info': {'link': '1', 'free_slots': 490, 'core': 0}},
                         self.routing.paths_list[0])
        self.assertEqual({'path': path2, 'link_info': {'link': '5', 'free_slots': 0, 'core': 0}},
                         self.routing.paths_list[1])

    def test_least_congested_path(self):
        """
        Tests the least congested path method. Force the simulation to choose a path by congesting the other one.
        """
        # Force simulation to pick a path (completely congest one path of the two)
        self.network_spec_db[('Lowell', 'Miami')]['cores_matrix'][0][0:] = 1
        self.network_spec_db[('Miami', 'Chicago')]['cores_matrix'][0][0:] = 1
        self.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix'][0][0:] = 1
        self.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix'][1][0:] = 1
        self.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix'][2][0:] = 1
        self.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix'][3][0:] = 1
        self.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix'][4][0:] = 1

        response = self.routing.least_congested_path()

        self.assertEqual(['Lowell', 'Las Vegas', 'Portland', 'San Francisco'], response, 'Incorrect path chosen.')

    def test_shortest_path(self):
        """
        Tests Dijkstra's shortest path algorithm.
        """
        self.routing.source = 'Lowell'
        self.routing.destination = 'Chicago'

        shortest_path, mod = self.routing.shortest_path()

        self.assertEqual(['Lowell', 'New York City', 'Richmond', 'Austin', 'Portland', 'San Francisco', 'Chicago'],
                         shortest_path)
