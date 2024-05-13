# pylint: disable=no-name-in-module
import os
import sys

import networkx as nx
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib import pyplot as plt

from data_scripts.structure_data import create_network
from gui.sim_thread.simulation_thread import SimulationThread


# TODO: Double check coding guidelines document:
#   - Assertive function names
#   - Complete docstrings
#   - Parameter types
class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class, central point that controls all GUI functionality
    and actions.
    """
    mw_main_view_widget = None  # this is container_widget
    mw_main_view_layout = None
    mw_main_view_splitter = None
    mw_main_view_left_splitter = None
    mw_main_view_right_splitter = None
    mw_topology_view_area = None

    def __init__(self):
        super().__init__()
        self.progress_bar = QtWidgets.QProgressBar()
        self.start_button = QtWidgets.QToolButton()
        self.pause_button = QtWidgets.QToolButton()
        self.stop_button = QtWidgets.QToolButton()
        self.simulation_thread = None
        self.network_option = ''
        self.init_mw_ui()

    def init_mw_ui(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle("SDNv1")
        self.resize(1280, 720)  # Set initial size of the window
        self.setStyleSheet("background-color: gray")
        self.center_window()
        self.init_mw_view_area()
        self.init_mw_menu_bar()
        self.init_mw_tool_bar()
        self.init_mw_status_bar()

    def init_mw_menu_bar(self):
        """
        Creates the menu bar.
        """
        # Create the menu bar
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("background-color: grey;")

        # Create File menu and add actions
        file_menu = menu_bar.addMenu('&File')
        open_action = QtWidgets.QAction('&Load Configuration from File', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Display topology information from File menu
        display_topology_action = QtWidgets.QAction('&Display topology', self)
        display_topology_action.triggered.connect(self.display_topology)
        file_menu.addAction(display_topology_action)

        save_action = QtWidgets.QAction('&Save', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        exit_action = QtWidgets.QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create Edit menu and add actions
        edit_menu = menu_bar.addMenu('&Edit')
        settings_action = QtWidgets.QAction('&Settings', self)
        settings_action.triggered.connect(self.open_settings)
        edit_menu.addAction(settings_action)

        # Create Help menu and add actions
        help_menu = menu_bar.addMenu('&Help')
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

    def init_mw_view_area(self):
        """
        Adds initial data displayed to the main screen, for example,
        the topology.
        """
        # Main container widget
        # this is needed because of mw central display
        self.mw_main_view_widget = QtWidgets.QWidget(self)
        # Set the color of the main container
        self.mw_main_view_widget.setStyleSheet(
            "background-color: #15b09e;"
        )

        # Layout for the container widget,
        # allowing for margins around the central data display
        self.mw_main_view_layout = QtWidgets.QHBoxLayout()
        # these margins to control the offset
        self.mw_main_view_layout.setContentsMargins(5, 5, 5, 5)
        self.mw_main_view_widget.setLayout(self.mw_main_view_layout)

        left_info_pane1 = QtWidgets.QWidget(self)
        left_info_pane1.setStyleSheet(
            "background-color: #545756;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )

        left_info_pane2 = QtWidgets.QWidget(self)
        left_info_pane2.setStyleSheet(
            "background-color: #545756;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )

        # initialize main view left splitter
        # main window splitters
        self.mw_main_view_left_splitter = QtWidgets.QSplitter()
        self.mw_main_view_left_splitter.setMinimumWidth(200)
        self.mw_main_view_left_splitter.setOrientation(QtCore.Qt.Vertical)
        self.mw_main_view_left_splitter.addWidget(left_info_pane1)
        self.mw_main_view_left_splitter.addWidget(left_info_pane2)

        self.mw_main_view_right_splitter = QtWidgets.QSplitter()
        self.mw_main_view_right_splitter.setOrientation(QtCore.Qt.Vertical)

        bottom_right_pane1 = QtWidgets.QWidget(self)
        bottom_right_pane1.setMinimumHeight(150)
        bottom_right_pane1.setMaximumHeight(200)
        bottom_right_pane1.setStyleSheet(
            "background-color: #545756;"
            "border-radius: 5px;"
            "border: 2px solid black;"
        )

        # scroll area for network topology
        self.mw_topology_view_area = QtWidgets.QScrollArea()
        self.mw_topology_view_area.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area.setStyleSheet(
            "background-color: white"
        )
        self.mw_topology_view_area.setWidget(init_topology_data)

        self.mw_main_view_right_splitter.addWidget(self.mw_topology_view_area)
        self.mw_main_view_right_splitter.addWidget(bottom_right_pane1)

        # create main window splitter
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

    def init_mw_tool_bar(self):
        """
        Adds controls to the toolbar.
        """
        # Create toolbar and add actions
        mw_toolbar = QtWidgets.QToolBar()
        self.addToolBar(QtCore.Qt.TopToolBarArea, mw_toolbar)
        mw_toolbar.setStyleSheet(
            "background-color: grey; color: white;")
        mw_toolbar.setMovable(False)
        mw_toolbar.setIconSize(QtCore.QSize(15, 15))
        mw_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        # path to play_button media file
        resource_name = "light-green-play-button.png"
        media_dir = "gui/media"
        self.start_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.start_button.setText("Start")
        self.start_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.start_button.setStyleSheet(
            "background-color: transparent;"
        )
        self.start_button.clicked.connect(self.start_simulation)

        # set up for pause button
        resource_name = "pause.png"
        self.pause_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.pause_button.setText("Pause")
        self.pause_button.setStyleSheet(
            "background-color: transparent;"
        )
        self.pause_button.clicked.connect(self.pause_simulation)

        # set up for stop button
        resource_name = "light-red-stop-button.png"
        self.stop_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.stop_button.setText("Stop")
        self.stop_button.setStyleSheet(
            "background-color: transparent;"
        )
        self.stop_button.clicked.connect(self.stop_simulation)

        settings_button = QtWidgets.QToolButton()
        resource_name = "gear.png"
        settings_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        settings_button.setText("Settings")
        settings_button.setStyleSheet(
            "background-color: transparent;"
        )
        settings_button.clicked.connect(self.open_settings)

        mw_toolbar.addSeparator()
        mw_toolbar.addWidget(self.start_button)
        mw_toolbar.addWidget(self.pause_button)
        mw_toolbar.addWidget(self.stop_button)
        mw_toolbar.addSeparator()
        mw_toolbar.addWidget(settings_button)

    def init_mw_status_bar(self):
        """
        Initializes the status bar.
        """
        main_status_bar = self.statusBar()
        main_status_bar.setStyleSheet(
            "background: gray;"
        )
        main_status_bar.addWidget(self.progress_bar)
        self.progress_bar.setVisible(False)

    def center_window(self):
        """
        Gets the center point of the window.
        """
        # Calculate the center point of the screen
        center_point = QtWidgets.QDesktopWidget().screenGeometry().center()
        # Reposition window in center of screen
        self.move(center_point - self.rect().center())

    def setup_simulation_thread(self):
        """
        Sets up one thread of the simulation.
        """
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.simulation_thread = SimulationThread()
        self.simulation_thread.progress_changed_sig.connect(
            self.update_progress
        )
        self.simulation_thread.finished_sig.connect(self.simulation_finished)
        self.simulation_thread.start()

    def start_simulation(self):
        """
        Begins the simulation.
        """
        if self.start_button.text() == "Resume":
            self.simulation_thread.resume()
            self.start_button.setText("Start")
        else:
            if (not self.simulation_thread or
                    not self.simulation_thread.isRunning()):
                self.setup_simulation_thread()
            else:
                self.simulation_thread.resume()
            self.start_button.setText("Start")

    def pause_simulation(self):
        """
        Pauses the simulation.
        """
        # print("Simulation paused")
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.pause()
            self.start_button.setText("Resume")

    def resume(self):
        """
        Resumes the simulation from a previous pause.
        """
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.pause()
            # Change button text to "Resume"
            self.start_button.setText(
                "Resume"
            )
        else:
            with QtCore.QMutexLocker(self.mutex):
                self.paused = False
            self.wait_cond.wakeAll()

    def stop_simulation(self):
        """
        Stops the simulation.
        """
        # print("Simulation stopped")
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
        self.start_button.setText("Start")

    # Placeholder methods for menu actions
    def open_file(self):
        """
        Opens a file.
        """
        # Set the file dialog to filter for .yml and .json files
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Configuration File", "", "Config Files (*.yml *.json)"
        )
        if file_name:
            print(f"Selected file: {file_name}")

    def display_topology(self):
        """
        Displays a network topology
        """
        network_selection_dialog = QtWidgets.QDialog()
        network_selection_dialog.setGeometry(100, 100, 100, 100)
        network_selection_dialog.adjustSize()
        network_selection_dialog.setSizeGripEnabled(True)

        # this centers the dialog box with respect to the main window
        dialog_pos = self.mapToGlobal(
            self.rect().center()) - network_selection_dialog.rect().center()
        network_selection_dialog.move(dialog_pos)

        network_selection_input = QtWidgets.QInputDialog(
            network_selection_dialog
        )
        items = ['USNet', 'NSFNet', 'Pan-European']
        item, ok = network_selection_input.getItem(
            network_selection_dialog, "Choose a network type:",
            "Select Network Type", items, 0, False
        )

        if ok and item:
            # set network_option to item
            self.network_option = item

    @staticmethod
    def save_file():
        """
        Saves a file.
        """
        print("Save file action triggered")

    @staticmethod
    def about():
        """
        Shows the About dialog.
        """
        print("Show about dialog")

    @staticmethod
    def open_settings():
        """
        Opens the settings panel.
        """
        print("Opening settings")

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
