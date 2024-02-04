import unittest
from unittest.mock import patch, MagicMock
from plot_scripts import plot_stats


class TestPlotStats(unittest.TestCase):
    """
    Class that contains methods to test plot_stats.py
    """

    def setUp(self):
        self.mock_sims_info_dict = {
            'networks_matrix': [['Network1', 'Network2']],
            'times_matrix': [['Time1', 'Time2']],
            'dates_matrix': [['Date1', 'Date2']],
        }

        self.plot_helpers_patch = patch('plot_scripts.plot_stats.PlotHelpers', autospec=True)
        self.MockPlotHelpers = self.plot_helpers_patch.start()  # pylint: disable=invalid-name
        self.mock_plot_helpers_instance = self.MockPlotHelpers.return_value
        self.mock_plot_helpers_instance.get_file_info = MagicMock()

        self.plot_stats = plot_stats.PlotStats(sims_info_dict=self.mock_sims_info_dict)

    def tearDown(self):
        self.plot_helpers_patch.stop()

    @patch('plot_scripts.plot_stats.plt')
    def test_setup_plot(self, mock_plt):
        """
        Tests setting up a plot.
        """
        title = "Test Title"
        y_lim = [0.1, 10]
        y_label = "Y Axis"
        x_label = "X Axis"

        # First call with all default parameters
        self.plot_stats._setup_plot(title, y_lim, y_label, x_label)  # pylint: disable=protected-access

        # Assertions for default call
        mock_plt.figure.assert_called_once_with(figsize=(7, 5), dpi=300)
        mock_plt.title.assert_called_once_with(f"{self.plot_stats.plot_props['title_names']} {title}")
        mock_plt.ylabel.assert_called_once_with(y_label)
        mock_plt.xlabel.assert_called_once_with(x_label)
        mock_plt.ylim.assert_called_with(0.1, 10)
        mock_plt.yscale.assert_called_once_with('log')
        mock_plt.xticks.assert_called_once()
        mock_plt.xlim.assert_called_once()
        mock_plt.grid.assert_called_once()

        mock_plt.reset_mock()
        self.plot_stats._setup_plot(title, y_lim, y_label, x_label, grid=False)  # pylint: disable=protected-access
        mock_plt.figure.assert_called_once_with(figsize=(7, 5), dpi=300)
        mock_plt.grid.assert_not_called()

    @patch('plot_scripts.plot_stats.plt')
    def test_plot_helper_one(self, mock_plt):
        """
        Tests plot helper one.
        """
        x_vals = 'x'
        y_vals_list = ['y1', 'y2']
        legend_val_list = ['legend1', 'legend2']
        legend_str = False
        file_name = 'test_plot.png'

        self.plot_stats.plot_props = {
            'title_names': 'Test Title',
            'style_list': ['-', '--'],
            'color_list': ['r', 'g'],
            'plot_dict': {
                'sim1': {
                    'info1': {
                        'x': [1, 2, 3],
                        'y1': [4, 5, 6],
                        'y2': [7, 8, 9],
                        'legend1': 'Legend 1',
                        'legend2': 'Legend 2'
                    }
                }
            },
            'x_tick_list': [1, 2, 3]
        }

        with patch.object(self.plot_stats, '_save_plot', autospec=True) as mock_save_plot:
            self.plot_stats._plot_helper_one(x_vals, y_vals_list, legend_val_list,  # pylint: disable=protected-access
                                             legend_str, file_name)
            mock_plt.plot.assert_called()
            mock_plt.legend.assert_called()
            mock_save_plot.assert_called_with(file_name=file_name)


if __name__ == '__main__':
    unittest.main()
