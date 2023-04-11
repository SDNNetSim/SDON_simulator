import unittest
import numpy as np
import networkx as nx

from sim_scripts.sdn_controller import SDNController


class TestSDNController(unittest.TestCase):
    """
    This class contains unit tests for methods found in the SDNController class.
    """

    def setUp(self):
        """
        Sets up this class.
        """
        self.req_id = 1
        self.net_spec_db = {
            (1, 2): {
                'cores_matrix': np.zeros((10, 100))
            },
            (2, 1): {
                'cores_matrix': np.zeros((10, 100))
            }
        }
        self.topology = nx.Graph()
        self.cores_per_link = 10
        self.path = [1, 2]
        self.sim_type = "yue"
        self.alloc_method = "first-fit"
        self.source = 1
        self.destination = 2
        self.mod_per_bw = {}
        self.chosen_bw = '100'
        self.max_slices = 1
        self.guard_slots = 0
        self.sdn_controller = SDNController(
            req_id=self.req_id,
            net_spec_db=self.net_spec_db,
            topology=self.topology,
            cores_per_link=self.cores_per_link,
            path=self.path,
            sim_type=self.sim_type,
            alloc_method=self.alloc_method,
            source=self.source,
            destination=self.destination,
            mod_per_bw=self.mod_per_bw,
            chosen_bw=self.chosen_bw,
            max_slices=self.max_slices,
            guard_slots=self.guard_slots
        )

    def test_release(self):
        """
        Tests the release method.
        """
        # Allocate the link in both directions
        self.net_spec_db[(1, 2)]['cores_matrix'][0][0] = self.req_id
        self.net_spec_db[(1, 2)]['cores_matrix'][0][1] = -self.req_id
        self.net_spec_db[(2, 1)]['cores_matrix'][0][0] = self.req_id
        self.net_spec_db[(2, 1)]['cores_matrix'][0][1] = -self.req_id

        self.sdn_controller.release()

        # Test forward direction
        self.assertTrue(np.array_equal(
            self.net_spec_db[(1, 2)]['cores_matrix'][0],
            np.zeros((100,))
        ))

        # Test backward direction
        self.assertTrue(np.array_equal(
            self.net_spec_db[(2, 1)]['cores_matrix'][0],
            np.zeros((100,))
        ))

    def test_allocate_with_guard_band(self):
        """
        Test the allocate method with a guard band.
        """
        self.sdn_controller.guard_slots = 1
        # Allocate five slots, guard band included
        self.sdn_controller.allocate(0, 5, 0)

        # Check actual request allocation
        self.assertTrue(np.all(self.net_spec_db[(1, 2)]['cores_matrix'][0][:4] == self.req_id))
        self.assertTrue(np.all(self.net_spec_db[(2, 1)]['cores_matrix'][0][:4] == self.req_id))

        # Check guard band
        self.assertEqual(self.net_spec_db[(1, 2)]['cores_matrix'][0][4], self.req_id * -1)
        self.assertEqual(self.net_spec_db[(2, 1)]['cores_matrix'][0][4], self.req_id * -1)

    def test_allocate_without_guard_band(self):
        """
        Test allocate without a guard band.
        """
        self.sdn_controller.guard_slots = 0
        self.sdn_controller.allocate(0, 5, 0)

        self.assertTrue(np.all(self.net_spec_db[(1, 2)]['cores_matrix'][0][:5] == self.req_id))
        self.assertTrue(np.all(self.net_spec_db[(2, 1)]['cores_matrix'][0][:5] == self.req_id))

        self.assertEqual(self.sdn_controller.net_spec_db[(1, 2)]['cores_matrix'][0][5], 0.0)
        self.assertEqual(self.sdn_controller.net_spec_db[(2, 1)]['cores_matrix'][0][5], 0.0)

    def test_allocate_conflict(self):
        """
        Test allocate when there is a spectrum utilization conflict.
        """
        self.net_spec_db[(1, 2)]['cores_matrix'][0][:6] = np.ones(6, dtype=np.float64)
        self.assertRaises(BufferError, self.sdn_controller.allocate, 0, 5, 0)
