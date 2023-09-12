import unittest
from sim_scripts.request_generator import generate


class TestGenerateFunction(unittest.TestCase):
    """
    This class contains unit tests for the `generate` function in the sim_scripts.request_generator module.
    """

    def test_generate_function(self):
        """
        Test that the `generate` function returns a dictionary with the correct keys, lengths, and
        number of requests for each bandwidth.
        """
        # Define inputs for generate function
        sim_type = 'yue'
        seed = 123
        nodes = ['A', 'B', 'C', 'D']
        hold_time_mean = 10.0
        arr_rate_mean = 2.0
        num_reqs = 100
        mod_per_bw = {'10': ['QPSK'], '100': ['16QAM']}
        req_dist = {'10': 0.2, '100': 0.8}

        # Call generate function with inputs
        result = generate(sim_type, seed, nodes, hold_time_mean, arr_rate_mean, num_reqs, mod_per_bw, req_dist)

        # Check that the output has the correct type
        self.assertIsInstance(result, dict)

        # Check that the output has the correct number of entries
        self.assertEqual(len(result), num_reqs * 2)

        # Check that each entry in the output has the correct keys
        expected_keys = ['id', 'source', 'destination', 'arrive', 'depart', 'request_type', 'bandwidth', 'mod_formats']
        for request in result.values():
            self.assertCountEqual(request.keys(), expected_keys)

        # Check that the number of requests for each bandwidth is correct
        num_10g_reqs = sum(request['bandwidth'] == '10' for request in result.values())
        self.assertEqual(num_10g_reqs / 2, int(num_reqs * req_dist['10']))

        num_100g_reqs = sum(request['bandwidth'] == '100' for request in result.values())
        self.assertEqual(num_100g_reqs / 2, int(num_reqs * req_dist['100']))
