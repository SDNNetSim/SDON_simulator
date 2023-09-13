import unittest
import numpy as np
import networkx as nx


class TestRouting(unittest.TestCase):
    """
    This class contains unit tests for methods found in the Routing class.
    """

    def setUp(self):
        """
        Sets up this class.
        """
        self.topology = nx.Graph()
        self.topology.add_edge(0, 1, free_slots=10, length=10)
        self.topology.add_edge(1, 2, free_slots=8, length=50)
        self.topology.add_edge(2, 3, free_slots=5, length=10)
        self.topology.add_edge(3, 4, free_slots=12, length=12)
        self.net_spec_db = {
            (0, 1): {'cores_matrix': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])},
            (1, 2): {'cores_matrix': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])},
            (2, 3): {'cores_matrix': np.array([[0, 0, 0], [0, 0, 0]])},
            (3, 4): {'cores_matrix': np.array([[0, 1, 0], [1, 0, 0]])}
        }
        self.mod_formats = {'QPSK': {'max_length': 150}, '16-QAM': {'max_length': 100}, '64-QAM': {'max_length': 50}}
        from sim_scripts.routing import Routing  # pylint: disable=import-outside-toplevel
        self.routing = Routing(0, 4, self.topology, self.net_spec_db, self.mod_formats, 5, 200)

    def test_find_least_cong_path(self):
        """
        Tests the least congested path method.
        """
        self.routing.paths_list = [
            {'path': [0, 1, 2, 3, 4], 'link_info': {'free_slots': 3}},
            {'path': [5, 6, 7, 8, 9], 'link_info': {'free_slots': 6}},
            {'path': [10, 11, 12, 13, 14], 'link_info': {'free_slots': 1}},
            {'path': [15, 16, 17, 18, 19], 'link_info': {'free_slots': 10}},
        ]
        least_cong_path = self.routing.find_least_cong_path()
        self.assertEqual([15, 16, 17, 18, 19], least_cong_path)

    def test_find_most_cong_link(self):
        """
        Tests the most congested link method. This method returns the number of free slots on the most occupied core,
        checking all links within the network.
        """
        # Test case 1: One link fully congested
        path = [0, 1, 2]
        self.assertEqual(0, self.routing.find_most_cong_link(path))

        # Test case 2: All links fully free
        path = [1, 2, 3]
        self.assertEqual(3, self.routing.find_most_cong_link(path))

        # Test case 3: Path with 3 links, mixed congestion
        path = [1, 2, 3, 4]
        self.assertEqual(2, self.routing.find_most_cong_link(path))

    def test_shortest_path(self):
        """
        Tests the shortest path method.
        """
        expected_path = [0, 1, 2, 3, 4]
        expected_mod_format = '16-QAM'
        path, mod_format = self.routing.least_weight_path(weight='length')

        self.assertEqual(path[0], expected_path)
        self.assertEqual(mod_format[0], expected_mod_format)


if __name__ == '__main__':
    unittest.main()
