import unittest
import numpy as np

from scripts.spectrum_assignment import SpectrumAssignment


# TODO: Add tests for all methods in spectrum_assignment.py


class TestSpectrumAssignment(unittest.TestCase):
    """
    Update
    """
    def setUp(self):
        """
        Update
        :return:
        """
        self.spec_assign = SpectrumAssignment()

        self.spec_assign.slots_needed = 100
        self.spec_assign.num_slots = 256

    def test_zeros_arr(self):
        """
        Update
        :return:
        """
        zeros_arr = np.zeros((1, self.spec_assign.num_slots))
        response = self.spec_assign.find_spectrum_slots(cores_matrix=zeros_arr)
        self.assertEqual(response, {'core_num': 0, 'start_slot': 0, 'end_slot': 99},
                         'Incorrect assignment received')

    def test_ones_arr(self):
        """
        Update
        :return:
        """
        ones_arr = np.ones((1, self.spec_assign.num_slots))
        response = self.spec_assign.find_spectrum_slots(cores_matrix=ones_arr)  # pylint: disable=line-too-long
        self.assertFalse(response,
                         'Assignment found in a core that has no available spectrum slots.')

    def test_one_core(self):
        """
        Update
        :return:
        """
        test_arr_one = np.zeros((1, self.spec_assign.num_slots))
        test_arr_two = np.zeros((1, self.spec_assign.num_slots))
        test_arr_three = np.zeros((1, self.spec_assign.num_slots))

        test_arr_one[0][33:75] = 1
        test_arr_two[0][0:20] = 1
        test_arr_three[0][0:156] = 1

        response_one = self.spec_assign.find_spectrum_slots(cores_matrix=test_arr_one)
        response_two = self.spec_assign.find_spectrum_slots(cores_matrix=test_arr_two)
        response_three = self.spec_assign.find_spectrum_slots(cores_matrix=test_arr_three)

        self.assertEqual(response_one, {'core_num': 0, 'start_slot': 75, 'end_slot': 174},
                         'Incorrect assignment received')
        self.assertEqual(response_two, {'core_num': 0, 'start_slot': 20, 'end_slot': 119},
                         'Incorrect assignment received')
        self.assertEqual(response_three, {'core_num': 0, 'start_slot': 156, 'end_slot': 255},
                         'Incorrect assignment received')

    def test_multiple_cores(self):
        """
        Update
        :return:
        """
        test_arr_one = np.zeros((5, self.spec_assign.num_slots))
        test_arr_two = np.zeros((5, self.spec_assign.num_slots))
        test_arr_two[0][0:] = 1
        test_arr_two[1][0:] = 1
        test_arr_two[2][0:] = 1
        test_arr_two[3][40:65] = 1
        test_arr_two[4][0:] = 1

        response_one = self.spec_assign.find_spectrum_slots(cores_matrix=test_arr_one)
        response_two = self.spec_assign.find_spectrum_slots(cores_matrix=test_arr_two)

        self.assertEqual(response_one, {'core_num': 0, 'start_slot': 0, 'end_slot': 99},
                         'Incorrect assignment received')
        self.assertEqual(response_two, {'core_num': 3, 'start_slot': 65, 'end_slot': 164},
                         'Incorrect assignment received')
