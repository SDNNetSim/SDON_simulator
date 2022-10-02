import unittest
import numpy as np

from scripts.spectrum_assignment import SpectrumAssignment


class TestSpectrumAssignment(unittest.TestCase):
    """
    Tests the spectrum assignment methods.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        path = ['Lowell', 'Boston', 'Miami', 'Chicago', 'San Francisco']
        network_spec_db = dict()

        for i in range(len(path) - 1):
            curr_tuple = (path[i], path[i + 1])
            num_cores = np.random.randint(1, 5)

            core_matrix = np.zeros((num_cores, 256))
            network_spec_db[curr_tuple] = core_matrix

        self.spec_assign = SpectrumAssignment(path=path, slots_needed=100, network_spec_db=network_spec_db)

    def test_free_spectrum(self):
        """
        Test where all spectrum slots are available.
        """
        response = self.spec_assign.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 0, 'end_slot': 99}, response,
                         'Incorrect assignment received')

    def test_full_spectrum(self):
        """
        Test where all spectrum slots in one link are full.
        """
        core_matrix = self.spec_assign.network_spec_db[('Chicago', 'San Francisco')]
        num_cores = np.shape(core_matrix)[0]

        for i in range(num_cores):
            core_matrix[i][0:] = 1

        response = self.spec_assign.find_free_spectrum()
        self.assertEqual(False, response)

    def test_forced_spectrum_assignment(self):
        """
        Make only one spectrum big enough available in one of the links, forcing the simulator to choose that one.
        """
        core_matrix = self.spec_assign.network_spec_db[('Miami', 'Chicago')]
        num_cores = np.shape(core_matrix)[0]

        for i in range(num_cores):
            if i == 0:
                core_matrix[i][0:50] = 1
                core_matrix[i][150:] = 1
            else:
                core_matrix[i][0:] = 1

        response = self.spec_assign.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 50, 'end_slot': 149}, response,
                         'Incorrect assignment received')


if __name__ == '__main__':
    unittest.main()
