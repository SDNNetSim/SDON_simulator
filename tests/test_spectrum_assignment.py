import unittest
import numpy as np

from sim_scripts.spectrum_assignment import SpectrumAssignment


class TestSpectrumAssignment(unittest.TestCase):
    """
    Tests the spectrum assignment methods.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        path = ['Lowell', 'Boston', 'Miami', 'Chicago', 'San Francisco']
        self.network_spec_db = dict()

        for i in range(len(path) - 1):
            curr_tuple = (path[i], path[i + 1])
            rev_curr_tuple = (path[i + 1], path[i])
            num_cores = 2

            core_matrix = np.zeros((num_cores, 256))
            self.network_spec_db[curr_tuple] = {}
            self.network_spec_db[rev_curr_tuple] = {}
            self.network_spec_db[curr_tuple]['cores_matrix'] = core_matrix
            self.network_spec_db[rev_curr_tuple]['cores_matrix'] = core_matrix

        self.spec_obj = SpectrumAssignment(path=path, slots_needed=100, network_spec_db=self.network_spec_db,
                                           guard_band=1)

    def test_free_spectrum_1(self):
        """
        Test where all spectrum slots are available with a request of size 1.
        """
        self.spec_obj.slots_needed = 1
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 0, 'end_slot': 2}, response,
                         'Incorrect assignment received')

    def test_free_spectrum_100(self):
        """
        Test where all spectrum slots are available with a request of size 100.
        """
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 0, 'end_slot': 101}, response,
                         'Incorrect assignment received')

    def test_full_spectrum(self):
        """
        Test where all spectrum slots in one link are full.
        """
        core_matrix = self.spec_obj.network_spec_db[('Chicago', 'San Francisco')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('San Francisco', 'Chicago')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]

        for i in range(num_cores):
            core_matrix[i][0:] = 1
            rev_core_matrix[i][0:] = 1

        response = self.spec_obj.find_free_spectrum()
        self.assertEqual(False, response)

    def test_forced_spectrum_assignment(self):
        """
        Make only one core able to allocate a request. Ensure the simulator chooses the correct one.
        """
        core_matrix = self.spec_obj.network_spec_db[('Miami', 'Chicago')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Chicago', 'Miami')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            if i == 0:
                core_matrix[i][0:50] = 1
                core_matrix[i][151:] = 1
                rev_core_matrix[i][0:50] = 1
                rev_core_matrix[i][151:] = 1
            else:
                core_matrix[i][0:] = 1
                rev_core_matrix[i][0:] = 1

        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 50, 'end_slot': 151}, response,
                         'Incorrect assignment received')

    def test_one_direction_free(self):
        """
        Test when all links from 'A' to 'B' full, but from 'B' to 'A' free.
        """
        core_matrix = self.spec_obj.network_spec_db[('Miami', 'Chicago')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][0:] = 1

        response = self.spec_obj.find_free_spectrum()
        self.assertEqual(False, response)

    def test_back_spectrum_1(self):
        """
        Test when the ending slots of a spectrum are free, the request is allocated properly for requests of size 1.
        """
        core_matrix = self.spec_obj.network_spec_db[('Miami', 'Chicago')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Chicago', 'Miami')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][:-2] = 1
            rev_core_matrix[i][:-2] = 1

        self.spec_obj.slots_needed = 1
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 254, 'end_slot': 256}, response,
                         'Incorrect assignment received for a request of size 1.')

    def test_back_spectrum_4(self):
        """
        Test when the ending slots of a spectrum are free, the request is allocated properly for requests of size 4.
        """
        core_matrix = self.spec_obj.network_spec_db[('Boston', 'Miami')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Miami', 'Boston')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][:-5] = 1
            rev_core_matrix[i][:-5] = 1

        self.spec_obj.slots_needed = 4
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 251, 'end_slot': 256}, response,
                         'Incorrect assignment received for a request of size 4.')

    def test_front_spectrum_1(self):
        """
        Test when the beginning slots of a spectrum are free, the request is allocated properly for requests of size 1.
        """
        core_matrix = self.spec_obj.network_spec_db[('Lowell', 'Boston')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Boston', 'Lowell')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][5:] = 1
            rev_core_matrix[i][5:] = 1

        self.spec_obj.slots_needed = 1
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 0, 'end_slot': 2}, response,
                         'Incorrect assignment received for a request of size 1.')

    def test_front_spectrum_4(self):
        """
        Test when the beginning slots of a spectrum are free, the request is allocated properly for requests of size 4.
        """
        core_matrix = self.spec_obj.network_spec_db[('Boston', 'Miami')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Miami', 'Boston')]['cores_matrix']

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][5:] = 1
            rev_core_matrix[i][5:] = 1

        self.spec_obj.slots_needed = 4
        response = self.spec_obj.find_free_spectrum()
        self.assertEqual({'core_num': 0, 'start_slot': 0, 'end_slot': 5}, response,
                         'Incorrect assignment received for a request of size4.')

    # TODO: Test different core allocations for best-fit
    def test_best_fit_1(self):
        """
        Test the best-fit spectrum allocation policy for a request of size 1.
        """
        self.spec_obj.best_fit = True
        self.spec_obj.slots_needed = 1
        core_matrix = self.spec_obj.network_spec_db[('Boston', 'Miami')]['cores_matrix']
        rev_core_matrix = self.spec_obj.network_spec_db[('Miami', 'Boston')]['cores_matrix']

        # Make three available windows, ensure the smallest one is picked
        window_one = [5, 50]
        window_two = [100, 103]
        window_three = [200, 250]

        num_cores = np.shape(core_matrix)[0]
        for i in range(num_cores):
            core_matrix[i][0:] = 1
            rev_core_matrix[i][0:] = 1

            # Open select windows
            core_matrix[i][window_one[0]:window_one[1]] = 0
            rev_core_matrix[i][window_one[0]:window_one[1]] = 0

            core_matrix[i][window_two[0]:window_two[1]] = 0
            rev_core_matrix[i][window_two[0]:window_two[1]] = 0

            core_matrix[i][window_three[0]:window_three[1]] = 0
            rev_core_matrix[i][window_three[0]:window_three[1]] = 0

        response = self.spec_obj.find_free_spectrum()
        print('bleep bloop')

    def test_best_fit_4(self):
        """
        Test the best-fit spectrum allocation policy for a request of size 4.
        """
        # TODO: Do the best fit policy on the first link here to see what happens
        self.spec_obj.best_fit = True
        self.spec_obj.slots_needed = 4


if __name__ == '__main__':
    unittest.main()
