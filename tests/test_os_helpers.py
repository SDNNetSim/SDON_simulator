import unittest
import os
from unittest.mock import patch, call

from helper_scripts.os_helpers import create_dir


class TestCreateDir(unittest.TestCase):
    """
    Tests the os_helpers script.
    """

    @patch('helper_scripts.os_helpers.os.makedirs')
    @patch('helper_scripts.os_helpers.os.path.abspath')
    @patch('helper_scripts.os_helpers.os.path.join')
    @patch('helper_scripts.os_helpers.os.path.basename')
    def test_create_dir_success(self, mock_basename, mock_join, mock_abspath, mock_makedirs):
        """
        Test that directories are created properly.
        """
        mock_abspath.return_value = '/abs/path/to'
        mock_join.side_effect = lambda a, _: a
        mock_basename.return_value = 'file'

        file_path = '/path/to/file'
        create_dir(file_path)

        # Check that abspath and join were called correctly
        mock_abspath.assert_called_once_with('/path/to/file')
        expected_join_calls = [call(file_path, os.pardir), call('/abs/path/to', 'file')]
        mock_join.assert_has_calls(expected_join_calls, any_order=False)

        # Check that makedirs was called correctly for both parent and last child directory
        expected_makedirs_calls = [call('/abs/path/to', exist_ok=True), call('/abs/path/to', exist_ok=True)]
        mock_makedirs.assert_has_calls(expected_makedirs_calls, any_order=False)

    def test_create_dir_with_none_path(self):
        """
        Test an invalid directory creation.
        """
        with self.assertRaises(ValueError):
            create_dir(None)


if __name__ == '__main__':
    unittest.main()
