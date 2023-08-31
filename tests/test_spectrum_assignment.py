import unittest
import numpy as np
from sim_scripts.spectrum_assignment import SpectrumAssignment


class TestSpectrumAssignment(unittest.TestCase):
    """
    This class contains unit tests for methods found in the SpectrumAssignment class.
    """

    def setUp(self):
        """
        Sets up this class.
        """
        self.net_spec_db = {(0, 1): {'cores_matrix': np.array([[0, 1, 0, 0, 0],
                                                               [1, 0, 1, 0, 0],
                                                               [0, 0, 0, 1, 0],
                                                               [0, 0, 0, 0, 0]])},
                            (1, 0): {'cores_matrix': np.array([[0, 1, 0, 0, 0],
                                                               [1, 0, 1, 0, 0],
                                                               [0, 0, 0, 1, 0],
                                                               [0, 0, 0, 0, 0]])},
                            (1, 2): {'cores_matrix': np.array([[1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 0],
                                                               [0, 1, 1, 1, 0],
                                                               [1, 0, 1, 0, 1]])},
                            (2, 1): {'cores_matrix': np.array([[1, 1, 1, 1, 1],
                                                               [1, 1, 1, 1, 0],
                                                               [0, 1, 1, 1, 0],
                                                               [1, 0, 1, 0, 1]])}
                            }

    def test_best_fit_allocation(self):
        """
        Tests the best_fit_allocation method.
        """
        spectrum_assignment = SpectrumAssignment([0, 1], 1, self.net_spec_db, 1)
        spectrum_assignment.cores_per_link = 4
        spectrum_assignment._best_fit_allocation()

        self.assertEqual(spectrum_assignment.response, {'core_num': 1, 'start_slot': 3, 'end_slot': 5})

    def test_check_other_links(self):
        """
        Tests the check_other_links method.
        """
        spectrum_assignment = SpectrumAssignment([0, 1, 2], 1, self.net_spec_db, 1)
        spectrum_assignment.cores_per_link = 4
        spectrum_assignment._check_other_links(1, 3, 5)

        self.assertFalse(spectrum_assignment.is_free)

    def test_first_fit_allocation(self):
        """
        Tests the first_fit_allocation method.
        """
        spectrum_assignment = SpectrumAssignment([0, 1], 1, self.net_spec_db, 1)
        spectrum_assignment.cores_per_link = 4
        spectrum_assignment.cores_matrix = self.net_spec_db[(0, 1)]['cores_matrix']
        spectrum_assignment._handle_first_last(flag='first_fit')

        self.assertEqual(spectrum_assignment.response, {'core_num': 0, 'start_slot': 2, 'end_slot': 4})
