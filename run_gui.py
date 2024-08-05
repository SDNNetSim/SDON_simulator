# pylint: disable=no-name-in-module
# pylint: disable=c-extension-no-member

import sys

from PyQt5 import QtWidgets, QtCore

from gui_scripts.gui_helpers.menu_helpers import MenuHelpers
from gui_scripts.gui_helpers.action_helpers import ActionHelpers
from gui_scripts.gui_helpers.button_helpers import ButtonHelpers


# TODO: Standards and guidelines regarding parameter types
class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class, central point that controls all GUI functionality and actions.
    """
    # TODO: Why define these here?
    mw_main_view_widget = None
    mw_main_view_layout = None
    mw_main_view_splitter = None
    mw_main_view_left_splitter = None
    mw_main_view_right_splitter = None
    mw_topology_view_area = None
    bottom_right_pane = None
    start_button = None
    pause_button = None
    stop_button = None

    def __init__(self):
        super().__init__()
        self.progress_bar = QtWidgets.QProgressBar()
        self.simulation_thread = None
        self.network_option = ''

        self.menu_help_obj = MenuHelpers()
        self.ac_help_obj = ActionHelpers()
        self.button_help_obj = ButtonHelpers()
        self.init_mw_ui()
        self.menu_bar_obj = None

        self.first_info_pane = None
        self.second_info_pane = None
        self.media_dir = None
        self.settings_button = None
        self.mw_toolbar = None

    def init_mw_ui(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle("SDNv1")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: gray")
        self.center_window()
        self.init_mw_view_area()
        self.init_mw_menu_bar()
        self.init_mw_tool_bar()
        self.init_mw_status_bar()

    # TODO: Calls an external class or something similar
    def init_mw_menu_bar(self):
        """
        Creates the menu bar.
        """
        self.menu_bar_obj = self.menuBar()
        self.menu_bar_obj.setStyleSheet("background-color: grey;")

        self.menu_help_obj.menu_bar_obj = self.menu_bar_obj
        self.menu_help_obj.create_file_menu()
        self.menu_help_obj.create_edit_menu()
        self.menu_help_obj.create_help_menu()

        self.ac_help_obj.mw_topology_view_area = self.mw_topology_view_area
        self.ac_help_obj.menu_help_obj = self.menu_help_obj
        self.ac_help_obj.menu_bar_obj = self.menu_bar_obj
        self.ac_help_obj.create_topology_action()
        self.ac_help_obj.create_save_action()
        self.ac_help_obj.create_exit_action()
        self.ac_help_obj.create_settings_action()
        self.ac_help_obj.create_about_action()

    # TODO: Comment and say these are to the left
    def _setup_first_info_pane(self):
        self.first_info_pane = QtWidgets.QWidget(self)

        left_info_pane1_layout = QtWidgets.QVBoxLayout()
        self.first_info_pane.setLayout(left_info_pane1_layout)
        left_info_pane1_init_data = QtWidgets.QLabel(
            "Nothing to display"
        )
        left_info_pane1_init_data.setStyleSheet(
            "border: 0px"
        )
        left_info_pane1_init_data.setAlignment(QtCore.Qt.AlignCenter)
        left_info_pane1_layout.addWidget(left_info_pane1_init_data)
        self.first_info_pane.setStyleSheet(
            "background-color: white;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )

    def _setup_second_info_pane(self):
        # left info pane 2 begin here
        self.second_info_pane = QtWidgets.QWidget(self)
        left_info_pane2_layout = QtWidgets.QVBoxLayout()
        self.second_info_pane.setLayout(left_info_pane2_layout)
        self.second_info_pane.setStyleSheet(
            "background-color: white;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )
        # initial data inside left info pane 2
        left_info_pane2_init_data = QtWidgets.QLabel(
            "Nothing to display"
        )
        left_info_pane2_init_data.setStyleSheet(
            "border: none"
        )
        left_info_pane2_init_data.setAlignment(QtCore.Qt.AlignCenter)
        # set layout with initial data
        left_info_pane2_layout.addWidget(left_info_pane2_init_data)

    def _init_left_splitter(self):
        self.mw_main_view_left_splitter = QtWidgets.QSplitter()
        self.mw_main_view_left_splitter.setMinimumWidth(200)
        self.mw_main_view_left_splitter.setOrientation(QtCore.Qt.Vertical)
        self.mw_main_view_left_splitter.addWidget(self.first_info_pane)
        self.mw_main_view_left_splitter.addWidget(self.second_info_pane)

    def _init_third_pane(self):
        self.bottom_right_pane = QtWidgets.QPlainTextEdit(self)
        self.bottom_right_pane.setReadOnly(True)
        self.bottom_right_pane.appendPlainText("No Data to display")
        self.bottom_right_pane.setMinimumHeight(150)
        self.bottom_right_pane.setMaximumHeight(200)
        self.bottom_right_pane.setStyleSheet(
            "background-color: white;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )

    def _init_topology(self):
        init_topology_data = QtWidgets.QLabel(
            "Nothing to display"
        )
        init_topology_data.setStyleSheet(
            "border: none"
        )
        init_topology_data.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area = QtWidgets.QScrollArea()
        self.mw_topology_view_area.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area.setStyleSheet(
            "background-color: white"
        )
        self.mw_topology_view_area.setWidget(init_topology_data)

    def _init_main_splitter(self):
        self.mw_main_view_splitter = QtWidgets.QSplitter()
        self.mw_main_view_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.mw_main_view_splitter.addWidget(
            self.mw_main_view_left_splitter
        )
        self.mw_main_view_splitter.addWidget(
            self.mw_main_view_right_splitter
        )

        self.mw_main_view_layout.addWidget(self.mw_main_view_splitter)

        # Set main window central widget
        self.setCentralWidget(self.mw_main_view_widget)

    def init_mw_view_area(self):
        """
        Adds initial data displayed to the main screen, for example, the topology.
        """
        self.mw_main_view_widget = QtWidgets.QWidget(self)
        self.mw_main_view_widget.setStyleSheet(
            "background-color: #15b09e;"
        )

        self.mw_main_view_layout = QtWidgets.QHBoxLayout()
        self.mw_main_view_layout.setContentsMargins(5, 5, 5, 5)
        self.mw_main_view_widget.setLayout(self.mw_main_view_layout)

        self._setup_first_info_pane()
        self._setup_second_info_pane()
        self._init_left_splitter()

        self.mw_main_view_right_splitter = QtWidgets.QSplitter()
        self.mw_main_view_right_splitter.setOrientation(QtCore.Qt.Vertical)

        self._init_third_pane()
        self._init_topology()

        self.mw_main_view_right_splitter.addWidget(self.mw_topology_view_area)
        self.mw_main_view_right_splitter.addWidget(self.bottom_right_pane)

        self._init_main_splitter()

    def _create_toolbar(self):
        self.mw_toolbar = QtWidgets.QToolBar()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.mw_toolbar)
        self.mw_toolbar.setStyleSheet("background-color: grey; color: white;")
        self.mw_toolbar.setMovable(False)
        self.mw_toolbar.setIconSize(QtCore.QSize(15, 15))
        self.mw_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

    def init_mw_tool_bar(self):
        """
        Adds controls to the toolbar.
        """
        self._create_toolbar()

        self.button_help_obj.bottom_right_pane = self.bottom_right_pane
        self.button_help_obj.progress_bar = self.progress_bar
        self.button_help_obj.create_settings_button()
        self.button_help_obj.create_start_button()
        self.button_help_obj.create_stop_button()
        self.button_help_obj.create_pause_button()

        self.mw_toolbar.addSeparator()
        self.mw_toolbar.addAction(self.button_help_obj.start_button)
        self.mw_toolbar.addAction(self.button_help_obj.pause_button)
        self.mw_toolbar.addAction(self.button_help_obj.stop_button)
        self.mw_toolbar.addSeparator()
        self.mw_toolbar.addWidget(self.button_help_obj.settings_button)

    def init_mw_status_bar(self):
        """
        Initializes the status bar.
        """
        main_status_bar = self.statusBar()
        main_status_bar.setStyleSheet("background: gray;")
        main_status_bar.addWidget(self.progress_bar)
        self.progress_bar.setVisible(False)

    def center_window(self):
        """
        Gets the center point of the window.
        """
        center_point = QtWidgets.QDesktopWidget().screenGeometry().center()  # Calculate the center point of the screen
        self.move(center_point - self.rect().center())  # Reposition window in the center of the screen

    @staticmethod
    def on_hover_change(label, data, hovered):
        """
        Change the display details based on a mouse hover.
        """
        if hovered:
            detailed_data = "<br>".join(f"{k}: {v}" for k, v in data.items())
            tooltip_text = f"Details:<br>{detailed_data}"
            label.setToolTip(tooltip_text)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
