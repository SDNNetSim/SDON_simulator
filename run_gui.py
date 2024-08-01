# pylint: disable=no-name-in-module
# pylint: disable=c-extension-no-member

import os
import sys

import networkx as nx
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib import pyplot as plt

from data_scripts.structure_data import create_network
from gui_scripts.gui_helpers.general_helpers import SettingsDialog, SimulationThread


# TODO: Instead of importing let's say, all action functions, have class inheritance instead with an action object.
class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class, central point that controls all GUI functionality and actions.
    """
    # TODO: Why define these here and not in INIT? This tells us they are constants.
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
        self.init_mw_ui()

        self.menu_bar_obj = None
        self.file_menu_obj = None
        self.edit_menu_obj = None
        self.help_menu_obj = None

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

    # TODO: Move to another file (menu helpers)
    def _create_file_menu(self):
        self.file_menu_obj = self.menu_bar_obj.addMenu('&File')
        open_action = QtWidgets.QAction('&Load Configuration from File', self)
        open_action.triggered.connect(self.open_file)
        self.file_menu_obj.addAction(open_action)

    # TODO: Add to standards and guidelines, must be called "create", if action must end in "action"
    # TODO: These one line functions can be added to the constructor?
    def _create_edit_menu(self):
        self.edit_menu_obj = self.menu_bar_obj.addMenu('&Edit')

    def _create_help_menu(self):
        self.help_menu_obj = self.menu_bar_obj.addMenu('&Help')

    # TODO: Move to another file (action helpers)
    def _create_topology_action(self):
        display_topology_action = QtWidgets.QAction('&Display topology', self)
        display_topology_action.triggered.connect(self.display_topology)
        self.file_menu_obj.addAction(display_topology_action)

    def _create_save_action(self):
        save_action = QtWidgets.QAction('&Save', self)
        save_action.triggered.connect(self.save_file)
        self.file_menu_obj.addAction(save_action)

    def _create_exit_action(self):
        exit_action = QtWidgets.QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        self.file_menu_obj.addAction(exit_action)

    def _create_settings_action(self):
        settings_action = QtWidgets.QAction('&Settings', self)
        settings_action.triggered.connect(self.open_settings)
        self.edit_menu_obj.addAction(settings_action)

    def _create_about_action(self):
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self.about)
        self.help_menu_obj.addAction(about_action)

    # TODO: Calls an external class or something similar
    def init_mw_menu_bar(self):
        """
        Creates the menu bar.
        """
        self.menu_bar_obj = self.menuBar()
        self.menu_bar_obj.setStyleSheet("background-color: grey;")

        self._create_file_menu()
        self._create_topology_action()
        self._create_save_action()
        self._create_exit_action()
        self._create_edit_menu()
        self._create_settings_action()
        self._create_help_menu()
        self._create_about_action()

    # TODO: Move to a window pane helpers?
    # TODO: Change names (pane1, pane2, left, right, etc.)
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

    def init_mw_tool_bar(self):
        """
        Adds controls to the toolbar.
        """
        self.start_button = QtWidgets.QAction()
        self.pause_button = QtWidgets.QAction()
        self.stop_button = QtWidgets.QAction()

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
        media_dir = os.path.join('gui_scripts', 'media')
        self.start_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.start_button.setText("Start")
        self.start_button.triggered.connect(self.start_simulation)

        # set up for pause button
        resource_name = "pause.png"
        self.pause_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.pause_button.setText("Pause")
        self.pause_button.triggered.connect(self.pause_simulation)

        # set up for stop button
        resource_name = "light-red-stop-button.png"
        self.stop_button.setIcon(
            QtGui.QIcon(os.path.join(os.getcwd(), media_dir, resource_name))
        )
        self.stop_button.setText("Stop")
        self.stop_button.triggered.connect(self.stop_simulation)

        # settings work now
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
        mw_toolbar.addAction(self.start_button)
        mw_toolbar.addAction(self.pause_button)
        mw_toolbar.addAction(self.stop_button)
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
        self.simulation_thread.output_hints_signal.connect(
            self.output_hints
        )
        self.simulation_thread.progress_changed.connect(
            self.update_progress
        )
        self.simulation_thread.finished_signal.connect(
            self.simulation_finished
        )
        self.simulation_thread.finished.connect(
            self.simulation_thread.deleteLater
        )
        self.simulation_thread.start()

    def output_hints(self, message):
        """
        Update.
        """
        self.bottom_right_pane1.appendPlainText(message)

    def start_simulation(self):
        """
        Begins the simulation.
        """
        if self.start_button.text() == "Resume":
            self.simulation_thread.resume()
            self.start_button.setText("Start")
        else:
            self.bottom_right_pane1.clear()
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
            with QtCore.QMutexLocker(self.simulation_thread.mutex):
                self.simulation_thread.paused = False
            self.simulation_thread.wait_cond.wakeOne()

    def stop_simulation(self):
        """
        Stops the simulation.
        """
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
            self.simulation_thread = None
        self.start_button.setText("Start")

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
        Displays a network topology.
        """
        network_selection_dialog = QtWidgets.QDialog()
        network_selection_dialog.setSizeGripEnabled(True)

        # this centers the dialog box with respect to the main window
        dialog_pos = self.mapToGlobal(
            self.rect().center()) - network_selection_dialog.rect().center()
        network_selection_dialog.move(dialog_pos)

        network_selection_input = QtWidgets.QInputDialog()
        # TODO: Hard coded
        # TODO: Does it update the configuration file? Or, does it just display it?
        items = ['USNet', 'NSFNet', 'Pan-European']
        item, is_ok = network_selection_input.getItem(
            network_selection_dialog, "Choose a network type:",
            "Select Network Type", items, 0, False
        )

        # TODO: No else statement for error checking, we should log things
        if is_ok and item:
            topology_information_dict = create_network(item)

            edge_list = [(src, des, {'weight': link_len})
                         for (src, des), link_len in
                         topology_information_dict.items()]
            network_topo = nx.Graph(edge_list)

            # graphing is done here
            figure = plt.figure()
            axis = figure.add_subplot(1, 1, 1)
            # spring_layout returns a dictionary of coordinates
            pos = nx.spring_layout(network_topo, seed=5, scale=3.5)
            nx.draw(network_topo, pos, with_labels=True, ax=axis, node_size=200,
                    font_size=8)
            # Close the matplotlib figure to prevent it from displaying
            plt.close(figure)

            figure.canvas.draw()
            width, height = figure.canvas.get_width_height()
            buffer = figure.canvas.buffer_rgba()
            image = QtGui.QImage(buffer, width, height,
                                 QtGui.QImage.Format_ARGB32)
            pixmap = QtGui.QPixmap.fromImage(image)

            # Display the QPixmap using a QLabel
            label = QtWidgets.QLabel(self)
            label.setFixedSize(pixmap.rect().size())
            # Center align pixmap, not even necessary (same size)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setPixmap(pixmap)

            self.mw_topology_view_area.setWidget(label)

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
