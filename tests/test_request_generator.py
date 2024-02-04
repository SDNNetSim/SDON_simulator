import unittest
import os
import json

from sim_scripts.request_generator import get_requests


class TestGetRequests(unittest.TestCase):

    def setUp(self):
        self.seed = 12345

        file_path = os.path.join('fixtures', 'engine_props.json')
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            self.engine_props = json.load(file_obj)

    def test_requests_length(self):
        """
        Test the number of requests generated.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        self.assertEqual(len(requests), self.engine_props['num_requests'] * 2)

    def test_no_duplicate_source_destination(self):
        """
        Test source and destination pairs.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        for key, value in requests.items():
            self.assertNotEqual(value['source'], value['destination'])

    def test_correct_bandwidth_distribution(self):
        """
        Tests the bandwidth distribution.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        bw_distribution = {bw: 0 for bw in self.engine_props['mod_per_bw']}
        for key, value in requests.items():
            if value['request_type'] == 'arrival':
                bw_distribution[value['bandwidth']] += 1
        for bw, count in self.engine_props['request_distribution'].items():
            expected_count = int(count * self.engine_props['num_requests'])
            self.assertEqual(bw_distribution[bw], expected_count)

    def test_arrival_departure_times(self):
        """
        Test arrival and departure times.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        for key, value in requests.items():
            if value['request_type'] == 'arrival':
                self.assertLess(value['arrive'], value['depart'])


if __name__ == '__main__':
    unittest.main()
