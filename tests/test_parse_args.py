import unittest
import argparse
from unittest.mock import patch

from config_scripts.parse_args import parse_args
from arg_scripts.config_args import COMMAND_LINE_PARAMS


class TestParseArgs(unittest.TestCase):
    """
    Test parse_args.py script.
    """
    maxDiff = None

    @staticmethod
    def generate_mock_value(arg_type: type, index):
        """
        Generate mock value based on argument type.
        """
        resp = None
        if arg_type == float:
            resp = float(index)
        elif arg_type == int:
            resp = int(index)
        elif arg_type == bool:
            resp = True
        elif arg_type == dict:
            resp = {f"key{index}": f"value{index}"}
        elif arg_type == list:
            resp = [f"value{index}"]
        else:
            resp = f"value{index}"

        return resp

    def generate_mock_args(self):
        """
        Generate mock arguments with an option to include the optimize flag.
        """
        mock_args = {param[0]: self.generate_mock_value(param[1], index) for index, param in
                     enumerate(COMMAND_LINE_PARAMS, 1)}

        return mock_args

    def generate_expected_args(self):
        """
        Generate expected arguments dictionary with an option to include the optimize flag.
        """
        expected_args = {param[0]: self.generate_mock_value(param[1], index) for index, param in
                         enumerate(COMMAND_LINE_PARAMS, 1)}
        return expected_args

    @patch('config_scripts.parse_args.argparse.ArgumentParser.parse_args')
    def test_parse_args_with_valid_arguments(self, mock_parse_args: object):
        """
        Test with valid arguments.
        """
        mock_parse_args.return_value = argparse.Namespace(**self.generate_mock_args())
        args = parse_args()
        parsed_args = self.generate_expected_args()
        self.assertEqual(args, parsed_args)

    @patch('config_scripts.parse_args.argparse.ArgumentParser.parse_args')
    def test_parse_args_with_missing_arguments(self, mock_parse_args: object):
        """
        Test with missing arguments.
        """
        mock_args = {param[0]: self.generate_mock_value(param[1], index) for index, param in
                     enumerate(COMMAND_LINE_PARAMS[:2], 1)}
        mock_parse_args.return_value = argparse.Namespace(**mock_args)
        args = parse_args()
        expected_args = {param[0]: self.generate_mock_value(param[1], index) for index, param in
                         enumerate(COMMAND_LINE_PARAMS[:2], 1)}

        self.assertEqual(args, expected_args)
        for param in COMMAND_LINE_PARAMS[2:]:
            self.assertNotIn(param[0], args)


if __name__ == '__main__':
    unittest.main()
