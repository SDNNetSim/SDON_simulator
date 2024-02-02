import unittest
import argparse
from unittest.mock import patch

from config_scripts.parse_args import parse_args


class TestParseArgs(unittest.TestCase):
    """
    Test parse_args.py script.
    """

    @patch('config_scripts.parse_args.argparse.ArgumentParser.parse_args')
    def test_parse_args_with_valid_arguments(self, mock_parse_args):
        """
        Test with valid arguments.
        """
        mock_parse_args.return_value = argparse.Namespace(argument1='value1', argument2=2)
        args = parse_args()
        self.assertEqual(args, {'argument1': 'value1', 'argument2': 2})

    @patch('config_scripts.parse_args.argparse.ArgumentParser.parse_args')
    def test_parse_args_with_missing_arguments(self, mock_parse_args):
        """
        Test with invalid arguments.
        """
        mock_parse_args.return_value = argparse.Namespace(argument1='value1')
        args = parse_args()

        self.assertEqual(args, {'argument1': 'value1'})
        self.assertNotIn('argument2', args)


if __name__ == '__main__':
    unittest.main()
