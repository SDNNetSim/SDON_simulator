# pylint: disable=no-name-in-module

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDesktopWidget,
    QWidget, QToolButton, QProgressBar, QAction,
    QFileDialog, QLabel, QVBoxLayout,
    QToolTip,
)
from PyQt5.QtGui import (
    QIcon, QFont,
)
from PyQt5.QtCore import (
    QSize, Qt, QMutexLocker,
)

from sim_thread.simulation_thread import SimulationThread
from labels.helper_labels import HoverLabel


# TODO: Double check coding guidelines document:
#   - Assertive function names
#   - Complete docstrings
#   - Parameter types
class MainWindow(QMainWindow):
    """
    The main window class, central point that controls all GUI functionality and actions.
    """

    def __init__(self):
        super().__init__()
        self.progress_bar = QProgressBar()
        self.start_button = QToolButton()
        self.pause_button = QToolButton()
        self.stop_button = QToolButton()
        self.simulation_thread = None
        self.initUI()

        self.hover_label = None
        self.paused = None

    def init_ui(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle("SDNv1")
        self.resize(1280, 720)  # Set initial size of the window
        self.setStyleSheet("background-color: #a3e1a4")  # Set light gray background color
        self.centerWindow()
        self.addCentralDataDisplay()
        self.addMenuBar()  # this adds the menubar
        self.addControlToolBar()
        self.initStatusBar()

    def add_menu_bar(self):
        """
        Creates the menu bar.
        """
        # Create the menu bar
        menu_bar = self.menuBar()

        # Create File menu and add actions
        file_menu = menu_bar.addMenu('&File')
        open_action = QAction('&Load Configuration from File', self)
        open_action.triggered.connect(self.openFile)
        file_menu.addAction(open_action)

        save_action = QAction('&Save', self)
        save_action.triggered.connect(self.saveFile)
        file_menu.addAction(save_action)

        exit_action = QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create Edit menu and add actions
        edit_menu = menu_bar.addMenu('&Edit')
        settings_action = QAction('&Settings', self)
        settings_action.triggered.connect(self.openSettings)
        edit_menu.addAction(settings_action)

        # Create Help menu and add actions
        help_menu = menu_bar.addMenu('&Help')
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

    def add_central_data_display(self):
        """
        Adds initial data displayed to the main screen, for example, the topology.
        """
        # Main container widget
        container_widget = QWidget()
        container_widget.setStyleSheet("background-color: #a3e1a4;")  # Set the color of the main container

        # Layout for the container widget, allowing for margins around the central data display
        container_layout = QVBoxLayout(container_widget)
        container_layout.setContentsMargins(10, 10, 10, 10)  # Adjust these margins to control the offset

        # The actual central data display widget with a white background
        data_display_widget = QWidget()
        data_display_widget.setStyleSheet(
            "background-color: white;"
            ""
        )

        self.hover_label = HoverLabel()
        # Assuming this is within your main window or a relevant container/widget
        self.hover_label.normalText = ""
        self.hover_label.hoverText = ""
        self.hover_label.setIcon(
            "/Users/kwadwoabempah/Python/GUI/pngtree-call-center-operator-with-phone-headset-icon-png-image_2059023.jpg")

        operator_status_info = {"ID": "n/a", "Region": "North America", "Connection Status": "Unknown"}

        # Connect the hoverChanged signal to a custom slot if you want to handle hover changes
        self.hoverLabel.hoverChanged.connect(
            lambda hovered: self.onHoverChange(self.hoverLabel, operator_status_info, hovered))

        container_layout.addWidget(data_display_widget)

        # Setting the container widget as the central widget of the main window
        self.setCentralWidget(container_widget)

        # Example content for the data display widget
        data_layout = QVBoxLayout(data_display_widget)
        data_label = QLabel("Application Data Display", data_display_widget)
        data_label.setAlignment(Qt.AlignCenter)
        # Add hoverLabel to the layout
        data_layout.addWidget(self.hoverLabel)
        data_layout.addWidget(data_label)

    def add_control_tool_bar(self):
        """
        Adds controls to the tool bar.
        """
        # Create toolbar and add actions
        toolbar = self.addToolBar("Simulation Controls")
        # Set gray background color and black text color for the toolbar
        toolbar.setStyleSheet("background-color: #d2dae2; color: white;")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))

        # Create custom tool button for Start action with transparent background
        # start_button = QToolButton()
        self.start_button.setIcon(QIcon("./media/light-green-play-button.png"))
        self.start_button.setText("Start")
        self.start_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.start_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
        self.start_button.clicked.connect(self.startSimulation)

        # set up for pause button
        # pause_button = QToolButton()
        self.pause_button.setIcon(QIcon("./media/pause.png"))
        self.pause_button.setText("Pause")
        self.pause_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.pause_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
        self.pause_button.clicked.connect(self.pauseSimulation)

        # set up for stop button
        # stop_button = QToolButton()
        self.stop_button.setIcon(QIcon("./media/light-red-stop-button.png"))
        self.stop_button.setText("Stop")
        self.stop_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.stop_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
        self.stop_button.clicked.connect(self.stopSimulation)

        settings_button = QToolButton()
        settings_button.setIcon(QIcon("./media/gear.png"))
        settings_button.setText("Settings")
        settings_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
        settings_button.clicked.connect(self.openSettings)

        toolbar.addSeparator()
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addSeparator()
        toolbar.addWidget(settings_button)

    def init_status_bar(self):
        """
        Initializes the status bar.
        """
        # Set green color
        self.statusBar().setStyleSheet(
            "QStatusBar { background-color: #333; color: white; }" +
            "QProgressBar::chunk { background-color: #4CAF50; }" +
            "QProgressBar { border: 2px solid grey; border-radius: 13px; text-align: right; color: black; background-color: #ddd;}"
        )
        self.progressBar.setStyleSheet("""
			QProgressBar {
				border: 2px solid grey;
				border-radius: 8px;  /* Rounds the corners of the progress bar */
				background-color: #ddd;
			}

			QProgressBar::chunk {
				background-color: #4CAF50;  /* Color of the progress chunks */
				margin: 0px; /* Optional: Adjusts the margin between chunks if needed */
				border-radius: 6px;  /* Rounds the corners of the progress chunks */
			}
		""")
        self.statusBar().addWidget(self.progressBar)
        self.progressBar.setVisible(False)

    def center_window(self):
        """
        Gets the center point of the window.
        """
        # Calculate the center point of the screen
        center_point = QDesktopWidget().screenGeometry().center()
        # Reposition window in center of screen
        self.move(center_point - self.rect().center())

    def setup_simulation_thread(self):
        """
        Sets up one thread of the simulation.
        """
        self.progressBar.setMaximum(1000)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)

        self.simulation_thread = SimulationThread()
        self.simulation_thread.progressChanged.connect(self.update_progress)
        self.simulation_thread.finished.connect(self.simulation_finished)
        self.simulation_thread.start()

    def start_simulation(self):
        """
        Begins the simulation.
        """
        if self.start_button.text() == "Resume":
            # print("Resuming simulation")
            self.simulation_thread.resume()
            self.start_button.setText("Start")
        else:
            # print("Starting simulation")
            if not self.simulation_thread or not self.simulation_thread.isRunning():
                self.setUpSimulationThread()
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
            self.start_button.setText("Resume")  # Change button text to "Resume"
        else:
            with QMutexLocker(self.mutex):
                self.paused = False
            self.wait_cond.wakeAll()

    def stop_simulation(self):
        """
        Stops the simulation.
        """
        # print("Simulation stopped")
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.progressBar.setValue(0)
            self.progressBar.setVisible(False)
        self.start_button.setText("Start")

    # Placeholder methods for menu actions
    def open_file(self):
        """
        Opens a file.
        """
        # Set the file dialog to filter for .yml and .json files
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Configuration File", "",
                                                   "Config Files (*.yml *.json)")
        if file_name:
            print(f"Selected file: {file_name}")
        # Here, you can add code to handle the opening and reading of the selected file

    @staticmethod
    def save_file():
        """
        Saves a file.
        """
        print("Save file action triggered")

    @staticmethod
    def about():
        """
        Shows the about dialog.
        """
        print("Show about dialog")

    @staticmethod
    def open_settings():
        """
        Opens the settings pannel.
        """
        print("Opening settings")

    def update_progress(self, value):
        """
        Updates the progress bar.
        """
        self.progressBar.setValue(value)

    def simulation_finished(self):
        """
        Finish the simulation.
        """
        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

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
    app = QApplication(sys.argv)

    # Set a custom font for tooltips if desired
    QToolTip.setFont(QFont('Arial', 10))
    app.setStyleSheet("""
		QToolTip {
			background-color: #f5f5f5;
			color: #333333;
			border: 1px solid #dcdcdc;
			padding: 4px;
			border-radius: 4px;
			opacity: 255;
		}
	""")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
