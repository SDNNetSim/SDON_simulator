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
        Pulls from configuration file to set up Controller Class.
        """
        self.req_id = 1
        self.net_spec_db = {
            (1, 2): {
                'cores_matrix': np.zeros((10, 100))
            },
            (2, 1): {
                'cores_matrix': np.zeros((10, 100))
            },
            (2, 3): {
                'cores_matrix': np.zeros((10, 100))
            },
            (3, 2): {
                'cores_matrix': np.zeros((10, 100))
            },
        }

        self.topology = nx.Graph()
        self.topology.add_edge(0, 1, free_slots=10, length=100)
        self.topology.add_edge(1, 2, free_slots=8, length=1000)
        self.topology.add_edge(2, 3, free_slots=5, length=10)
        self.topology.add_edge(3, 4, free_slots=12, length=12)

        self.cores_per_link = 10
        self.path = [1, 2]
        self.sim_type = "yue"
        self.alloc_method = "first_fit"
        self.source = 1
        self.destination = 2
        self.mod_per_bw = {str(bw): {mod: {"max_length": 10, "slots_needed": 1} for mod in ["QPSK", "16-QAM", "64-QAM"]}
                           for bw in [25, 50]}
        self.chosen_bw = '100'
        self.max_segments = 1
        self.guard_slots = 0

        # TODO: Make more efficient
        properties = {
            'topology': self.topology,
            'topology_info': self.topology,
            'cores_per_link': self.cores_per_link,
            'sim_type': self.sim_type,
            'allocation_method': self.alloc_method,
            'route_method': 'shortest_path',
            'dynamic_lps': False,
            'ai_algorithm': None,
            'beta': 0.5,
            'max_segments': self.max_segments,
            'guard_slots': self.guard_slots,
            'mod_per_bw': self.mod_per_bw,
            'spectral_slots': 100,
            'bw_per_slot': 12.5,
            'input_power': None,
            'egn_model': False,
            'phi': None,
            'bi_directional': False,
            'xt_noise': False,
            'requested_xt': -30,
            'check_snr': False
        }
        self.sdn_controller = SDNController(properties=properties)
        self.sdn_controller.req_id = self.req_id
        self.sdn_controller.net_spec_db = self.net_spec_db
        self.sdn_controller.path = self.path
        self.sdn_controller.chosen_bw = self.chosen_bw
        self.sdn_controller.source = self.source
        self.sdn_controller.destination = self.destination
        self.sdn_controller.topology = self.topology

    def test_release(self):
        """
        Tests the release method.

        :return: None
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

        :return: None
        """
        self.sdn_controller.sdn_props['guard_slots'] = 1
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

        :return: None
        """
        self.sdn_controller.guard_slots = 0
        self.sdn_controller.allocate(0, 5, 0)

        self.assertTrue(np.all(self.net_spec_db[(1, 2)]['cores_matrix'][0][:5] == self.req_id))
        self.assertTrue(np.all(self.net_spec_db[(2, 1)]['cores_matrix'][0][:5] == self.req_id))

        self.assertEqual(self.sdn_controller.net_spec_db[(1, 2)]['cores_matrix'][0][6], 0.0)
        self.assertEqual(self.sdn_controller.net_spec_db[(2, 1)]['cores_matrix'][0][6], 0.0)

    def test_allocate_conflict(self):
        """
        Test allocate when there is a spectrum utilization conflict.

        :return: None
        """
        self.net_spec_db[(1, 2)]['cores_matrix'][0][:6] = np.ones(6, dtype=np.float64)
        self.assertRaises(BufferError, self.sdn_controller.allocate, 0, 5, 0)

    def test_allocate_lps_with_unsuccessful_lps(self):
        """
        Test the allocate_lps method when we have a bandwidth of 25 or maximum allowed slicing equal to one.

        :return: None
        """
        self.sdn_controller.chosen_bw = '25'
        self.sdn_controller.max_slices = 2
        result = self.sdn_controller.allocate_lps()
        self.assertFalse(result)

        self.sdn_controller.chosen_bw = '400'
        self.sdn_controller.max_slices = 1
        result = self.sdn_controller.allocate_lps()
        self.assertFalse(result)

    def test_unsuccessful_dynamic_lps(self):
        """
        Test allocate_dynamic_lps to check for an unsuccessful light path slice.

        :return: None
        """
        self.sdn_controller.chosen_bw = '100'
        self.sdn_controller.max_slices = 10
        self.net_spec_db[(2, 3)]['cores_matrix'] = np.ones((10, 100))
        self.net_spec_db[(3, 2)]['cores_matrix'] = np.ones((10, 100))
        result = self.sdn_controller.allocate_lps()
        self.assertFalse(result)

        self.sdn_controller.chosen_bw = '100'
        self.sdn_controller.max_segments = 1
        result = self.sdn_controller.allocate_lps()
        self.assertFalse(result)
