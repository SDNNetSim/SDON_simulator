import unittest
import numpy as np

from scripts.sdn_controller import release


class TestSDNController(unittest.TestCase):
    def setUp(self):
        self.single_core_arr = np.zeros((1, 256))
        self.multi_core_arr = np.zeros((5, 256))

        self.single_core_db = {('Lowell', 'Boston'): self.single_core_arr}
        self.multi_core_db = {('Lowell', 'Boston'): self.multi_core_arr}
        self.path = ['Lowell', 'Boston']

    def test_single_core_release(self):
        slot_num = 50
        num_occ_slots = 100
        self.single_core_db[('Lowell', 'Boston')][slot_num:num_occ_slots - 1] = 1

        response = release(network_spec_db=self.single_core_db, path=self.path,
                           slot_num=slot_num, num_occ_slots=num_occ_slots)

        self.assertEqual(response[('Lowell', 'Boston')][0].all(), 0,
                         'Single core fiber was not released properly.')

    def test_multi_core_release(self):
        core_num = 4
        slot_num = 110
        num_occ_slots = 50
        self.multi_core_db[('Lowell', 'Boston')][core_num][slot_num:num_occ_slots] = 1

        response = release(network_spec_db=self.multi_core_db, path=self.path, slot_num=slot_num,
                           num_occ_slots=num_occ_slots)

        self.assertEqual(response[('Lowell', 'Boston')][core_num].all(), 0,
                         'Multi core fiber was not released properly.')
