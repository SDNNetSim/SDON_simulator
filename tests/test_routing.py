import unittest
import numpy as np

from scripts.routing import Routing
from scripts.engine import Engine


class TestRouting(unittest.TestCase):
    """
    Tests the routing methods.
    """

    def setUp(self):
        """
        Sets up the class for testing.
        """
        self.engine = Engine(sim_input_fp='../tests/data/input2.json')
        self.engine.load_input()
        self.engine.create_pt()

        self.physical_topology = self.engine.physical_topology
        self.network_spec_db = self.engine.network_spec_db

    def test_find_least_cong_route(self):
        pass

    def test_find_most_cong_link(self):
        pass

    def test_least_congested_path(self):
        pass
