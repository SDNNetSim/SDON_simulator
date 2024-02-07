import unittest
import os
from config_scripts.setup_config import read_config
from config_scripts.parse_args import parse_args


class TestReadConfig(unittest.TestCase):
    """
    Tests setup_config.py
    """

    def setUp(self):
        self.valid_conf = os.path.join('..', 'tests', 'fixtures', 'valid_config.ini')
        self.invalid_conf = os.path.join('..', 'tests', 'fixtures', 'invalid_config.ini')
        self.args_obj = parse_args()

    def test_successful_config_read(self):
        """
        Test successful configuration file.
        """
        config_dict = read_config(self.args_obj, self.valid_conf)
        self.assertIsNotNone(config_dict)

    def test_missing_config_file(self):
        """
        Test unsuccessful configuration file.
        """
        with self.assertRaises(ValueError):
            read_config(self.args_obj, 'None')

    def test_invalid_config_read(self):
        """
        Test invalid configuration file.
        """
        with self.assertRaises(ValueError) as context:
            read_config(self.args_obj, self.invalid_conf)
        self.assertIn("Missing 'route_method' in the general_settings section", str(context.exception))

    def test_command_line_input(self):
        """
        Test command line input.
        """
        self.args_obj['holding_time'] = 1000.0
        config_dict = read_config(self.args_obj, self.valid_conf)

        self.assertEqual(config_dict['s1']['holding_time'], 1000.0)
