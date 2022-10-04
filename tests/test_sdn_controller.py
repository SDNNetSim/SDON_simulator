import unittest
import numpy as np

from scripts.sdn_controller import release


class TestSDNController(unittest.TestCase):
    """
    Tests the SDN controller module.
    """

    def setUp(self):
        """
        Set up the class for testing.
        """
        self.path = ['Lowell', 'Boston', 'Miami', 'Chicago', 'San Francisco']
        self.network_spec_db = dict()

        for i in range(len(self.path) - 1):
            curr_tuple = (self.path[i], self.path[i + 1])
            num_cores = np.random.randint(1, 5)

            core_matrix = np.zeros((num_cores, 512))
            self.network_spec_db['path'] = curr_tuple
            self.network_spec_db['cores_matrix'] = core_matrix

    def test_release(self):
        """
        Test that a link has been released of its resources properly.
        """
        for i in range(len(self.path) - 1):
            curr_tuple = (self.path[i], self.path[i + 1])
            self.network_spec_db[curr_tuple]['cores_matrix'][0][50:100] = 1

        response = release(network_spec_db=self.network_spec_db, path=self.path, start_slot=50, num_slots=50,
                           core_num=0)

        for nodes, link in response.items():  # pylint: disable=unused-variable
            self.assertNotEqual(1, link[0].any(), 'Spectrum slots were not released correctly.')


if __name__ == '__main__':
    unittest.main()
