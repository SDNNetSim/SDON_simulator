import unittest
import networkx as nx
import numpy as np
from sim_scripts.routing import Routing


class TestRouting(unittest.TestCase):
    """
    This class tests the sim_scripts.routing module.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.routing = Routing()
        self.topology = nx.Graph()
        self.topology.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)])
        self.net_spec_db = {'link1': {'free_slots': 100, 'mod_format': 'QAM16'},
                            'link2': {'free_slots': 50, 'mod_format': 'QAM8'},
                            'link3': {'free_slots': 10, 'mod_format': 'BPSK'}}
        self.mod_formats = {'QAM16': 100, 'QAM8': 50, 'BPSK': 10}

    def test_find_least_cong_route(self):
        """
        Test find_least_cong_route() method with a non-empty path list.
        """
        routing = Routing(1, 4, self.topology, self.net_spec_db, self.mod_formats, slots_needed=5, bandwidth=10)
        routing.paths_list = [{'path': [1, 2, 3, 4], 'link_info': self.net_spec_db['link1']},
                              {'path': [1, 3, 4], 'link_info': self.net_spec_db['link2']},
                              {'path': [1, 2, 4], 'link_info': self.net_spec_db['link3']}]
        expected_output = [1, 2, 3, 4]
        self.assertEqual(routing.find_least_cong_route(), expected_output)

    def test_find_least_cong_route_with_empty_path_list(self):
        """
        Test find_least_cong_route() method with an empty path list.
        """
        routing = Routing(1, 4, self.topology, self.net_spec_db, self.mod_formats, slots_needed=5, bandwidth=10)
        routing.paths_list = []
        with self.assertRaises(IndexError):
            routing.find_least_cong_route()

    def test_find_most_cong_link(self):
        """
        Test the find_most_cong_link() method.
        """
        # Test with a path where all links have the same amount of congestion
        path = [0, 1, 2, 3]
        self.routing.net_spec_db = {(0, 1): {'cores_matrix': np.array([[1, 1, 0], [0, 1, 0]])},
                                    (1, 2): {'cores_matrix': np.array([[0, 0, 1], [1, 0, 0]])},
                                    (2, 3): {'cores_matrix': np.array([[0, 0, 0], [1, 1, 0]])}}
        self.assertEqual(self.routing.find_most_cong_link(path), 0)

        # Test with a path where one link is more congested than the others
        path = [0, 1, 2, 3]
        self.routing.net_spec_db = {(0, 1): {'cores_matrix': np.array([[1, 0, 0], [1, 0, 0]])},
                                    (1, 2): {'cores_matrix': np.array([[0, 0, 1], [1, 1, 0]])},
                                    (2, 3): {'cores_matrix': np.array([[0, 0, 0], [0, 1, 0]])}}
        self.assertEqual(self.routing.find_most_cong_link(path), 1)

        # Test with a path where all links are fully congested
        path = [0, 1, 2, 3]
        self.routing.net_spec_db = {(0, 1): {'cores_matrix': np.array([[1, 1, 1], [1, 1, 1]])},
                                    (1, 2): {'cores_matrix': np.array([[1, 1, 1], [1, 1, 1]])},
                                    (2, 3): {'cores_matrix': np.array([[1, 1, 1], [1, 1, 1]])}}
        self.assertEqual(self.routing.find_most_cong_link(path), 0)
