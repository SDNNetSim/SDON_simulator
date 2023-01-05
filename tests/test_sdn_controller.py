import unittest
import numpy as np

from scripts.sdn_controller import handle_arrive_rel


class TestSDNController(unittest.TestCase):
    """
    Tests the SDN controller module.
    """

    def setUp(self):
        """
        Set up the class for testing.
        """
        self.path = ['Lowell', 'Miami', 'Chicago', 'San Francisco']
        self.network_spec_db = dict()

        for i in range(len(self.path) - 1):
            curr_tuple = (self.path[i], self.path[i + 1])
            rev_tuple = (self.path[i + 1], self.path[i])
            num_cores = np.random.randint(1, 5)

            core_matrix = np.zeros((num_cores, 512))
            self.network_spec_db[curr_tuple] = {}
            self.network_spec_db[rev_tuple] = {}
            self.network_spec_db[curr_tuple]['cores_matrix'] = core_matrix
            self.network_spec_db[rev_tuple]['cores_matrix'] = core_matrix

    def test_release(self):
        """
        Test that a link has been released of its resources properly.
        """
        for i in range(len(self.path) - 1):
            curr_tuple = (self.path[i], self.path[i + 1])

            self.network_spec_db[curr_tuple]['cores_matrix'][0][50:100] = 1

        response = handle_arrive_rel(network_spec_db=self.network_spec_db, path=self.path, start_slot=50, num_slots=50,
                                     core_num=0, req_type='release')

        test_matrix = response[('Lowell', 'Miami')]['cores_matrix'][0]
        test_rev_matrix = response[('Miami', 'Lowell')]['cores_matrix'][0]
        self.assertEqual({0.0}, set(test_matrix),
                         'Spectrum slots were not released correctly from nodes Lowell to Miami.')
        self.assertEqual({0.0}, set(test_rev_matrix),
                         'Spectrum slots were not released correctly from nodes Miami to Lowelll.')

    def test_arrival(self):
        """
        Test that a link has allocated its resources properly.
        """
        response = handle_arrive_rel(network_spec_db=self.network_spec_db, path=self.path, start_slot=50, num_slots=50,
                                     core_num=0, req_type='arrival')

        for i in range(len(self.path) - 1):
            curr_array = response[(self.path[i], self.path[i + 1])]['cores_matrix'][0][50:100]
            rev_array = response[(self.path[i + 1], self.path[i])]['cores_matrix'][0][50:100]

            test_set = set(curr_array)
            test_rev_set = set(rev_array)

            self.assertEqual({1}, test_set,
                             f'Resources not allocated properly from nodes {self.path[i]} to {self.path[i + 1]}')
            self.assertEqual({1}, test_rev_set,
                             f'Resources not allocated properly from nodes {self.path[i + 1]} to {self.path[i]}')


if __name__ == '__main__':
    unittest.main()
