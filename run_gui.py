# pylint: disable=no-name-in-module
# pylint: disable=c-extension-no-member

import sys

from PyQt5 import QtWidgets, QtCore

from gui_scripts.gui_helpers.general_helpers import SettingsDialog
from gui_scripts.gui_helpers.menu_helpers import MenuHelpers
from gui_scripts.gui_helpers.action_helpers import ActionHelpers


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
    bottom_right_pane1 = None
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
        self.bottom_right_pane1 = QtWidgets.QPlainTextEdit(self)
        self.bottom_right_pane1.setReadOnly(True)
        self.bottom_right_pane1.appendPlainText("No Data to display")
        self.bottom_right_pane1.setMinimumHeight(150)
        self.bottom_right_pane1.setMaximumHeight(200)
        self.bottom_right_pane1.setStyleSheet(
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
        self.mw_main_view_right_splitter.addWidget(self.bottom_right_pane1)

        self._init_main_splitter()

    # def _create_start_button(self):
    #     self.start_button = QtWidgets.QAction()
    #     resource_name = "light-green-play-button.png"
    #     self.media_dir = os.path.join('gui_scripts', 'media')
    #     self.start_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
    #     self.start_button.setText("Start")
    #     self.start_button.triggered.connect(self.start_simulation)
    #
    # def _create_pause_button(self):
    #     self.pause_button = QtWidgets.QAction()
    #     resource_name = "pause.png"
    #     self.pause_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
    #     self.pause_button.setText("Pause")
    #     self.pause_button.triggered.connect(self.pause_simulation)
    #
    # def _create_stop_button(self):
    #     self.stop_button = QtWidgets.QAction()
    #     resource_name = "light-red-stop-button.png"
    #     self.stop_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
    #     self.stop_button.setText("Stop")
    #     self.stop_button.triggered.connect(self.stop_simulation)
    #
    # def _create_settings_button(self):
    #     self.settings_button = QtWidgets.QToolButton()
    #     resource_name = "gear.png"
    #     self.settings_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
    #     self.settings_button.setText("Settings")
    #     self.settings_button.setStyleSheet("background-color: transparent;")
    #     self.settings_button.clicked.connect(self.open_settings)

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
        # self._create_start_button()
        # self._create_pause_button()
        # self._create_stop_button()
        # self._create_settings_button()

        self.mw_toolbar.addSeparator()
        self.mw_toolbar.addAction(self.start_button)
        self.mw_toolbar.addAction(self.pause_button)
        self.mw_toolbar.addAction(self.stop_button)
        self.mw_toolbar.addSeparator()
        self.mw_toolbar.addWidget(self.settings_button)

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

    # def setup_simulation_thread(self):
    #     """
    #     Sets up one thread of the simulation.
    #     """
    #     self.progress_bar.setMaximum(1000)
    #     self.progress_bar.setValue(0)
    #     self.progress_bar.setVisible(True)
    #
    #     self.simulation_thread = SimulationThread()
    #     self.simulation_thread.output_hints_signal.connect(self.output_hints)
    #     self.simulation_thread.progress_changed.connect(self.update_progress)
    #     self.simulation_thread.finished_signal.connect(self.simulation_finished)
    #     self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
    #     self.simulation_thread.start()

    def output_hints(self, message):
        """
        Outputs hints.
        """
        self.bottom_right_pane1.appendPlainText(message)

    # def start_simulation(self):
    #     """
    #     Begins the simulation.
    #     """
    #     if self.start_button.text() == "Resume":
    #         self.simulation_thread.resume()
    #         self.start_button.setText("Start")
    #     else:
    #         self.bottom_right_pane1.clear()
    #         if (not self.simulation_thread or
    #                 not self.simulation_thread.isRunning()):
    #             self.setup_simulation_thread()
    #         else:
    #             self.simulation_thread.resume()
    #         self.start_button.setText("Start")

    # def pause_simulation(self):
    #     """
    #     Pauses the simulation.
    #     """
    #     if self.simulation_thread and self.simulation_thread.isRunning():
    #         self.simulation_thread.pause()
    #         self.start_button.setText("Resume")
    #
    # def resume(self):
    #     """
    #     Resumes the simulation from a previous pause.
    #     """
    #     if self.simulation_thread and self.simulation_thread.isRunning():
    #         self.simulation_thread.pause()
    #         self.start_button.setText("Resume")
    #     else:
    #         with QtCore.QMutexLocker(self.simulation_thread.mutex):
    #             self.simulation_thread.paused = False
    #         self.simulation_thread.wait_cond.wakeOne()
    #
    # def stop_simulation(self):
    #     """
    #     Stops the simulation.
    #     """
    #     if self.simulation_thread and self.simulation_thread.isRunning():
    #         self.simulation_thread.stop()
    #         self.progress_bar.setValue(0)
    #         self.progress_bar.setVisible(False)
    #         self.simulation_thread = None
    #     self.start_button.setText("Start")

    def update_progress(self, value):
        """
        Updates the progress bar.
        """
        self.progress_bar.setValue(value)

    def simulation_finished(self):
        """
        Finish the simulation.
        """
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.simulation_thread = None

    @staticmethod
    def open_settings():
        """
        Opens the settings panel.
        """
        settings_dialog = SettingsDialog()
        settings_dialog.setModal(True)
        settings_dialog.setStyleSheet("""
            background-color: white;
            background-color: white;
        """)
        if settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            print(settings_dialog.get_settings())

    @staticmethod
    def on_hover_change(label, data, hovered):
        """
        Change the display details based on a mouse hover.
        """
        if hovered:
            detailed_data = "<br>".join(f"{k}: {v}" for k, v in data.items())
            tooltip_text = f"Details:<br>{detailed_data}"
            # print(f"Setting tooltip: {tooltipText}")  # Debug print
            label.setToolTip(tooltip_text)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
