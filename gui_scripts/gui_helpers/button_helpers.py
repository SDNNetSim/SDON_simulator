# pylint: disable=c-extension-no-member

import os

from PyQt5 import QtWidgets, QtGui, QtCore

from gui_scripts.gui_helpers.general_helpers import SimulationThread
from gui_scripts.gui_helpers.general_helpers import SettingsDialog


class ButtonHelpers(QtCore.QObject):
    """
    Contains methods related to setting up the buttons and their potential options.
    """

    def __init__(self):
        self.bottom_right_pane = None
        self.progress_bar = None

        self.start_button = None
        self.pause_button = None
        self.stop_button = None
        self.settings_button = None
        self.simulation_thread = None

        self.media_dir = 'media'

    def output_hints(self, message: str):
        """
        Outputs hints.
        """
        self.bottom_right_pane.appendPlainText(message)

    def update_progress(self, value: float):
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

    def setup_simulation_thread(self):
        """
        Sets up one thread of the simulation.
        """
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.simulation_thread = SimulationThread()
        self.simulation_thread.output_hints_signal.connect(self.output_hints)
        self.simulation_thread.progress_changed.connect(self.update_progress)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.simulation_thread.start()

    def start_simulation(self):
        """
        Begins the simulation.
        """
        if self.start_button.text() == "Resume":
            self.simulation_thread.resume()
            self.start_button.setText("Start")
        else:
            self.bottom_right_pane.clear()
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
            self.start_button.setText("Resume")
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

    def create_start_button(self):
        """
        Creates the start button and action.
        """
        self.start_button = QtWidgets.QAction()
        resource_name = "light-green-play-button.png"
        self.media_dir = os.path.join('gui_scripts', 'media')
        self.start_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.start_button.setText("Start")
        self.start_button.triggered.connect(self.start_simulation)

    def create_pause_button(self):
        """
        Creates the pause button and action.
        """
        self.pause_button = QtWidgets.QAction()
        resource_name = "pause.png"
        self.pause_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.pause_button.setText("Pause")
        self.pause_button.triggered.connect(self.pause_simulation)

    def create_stop_button(self):
        """
        Creates the stop button and action.
        """
        self.stop_button = QtWidgets.QAction()
        resource_name = "light-red-stop-button.png"
        self.stop_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.stop_button.setText("Stop")
        self.stop_button.triggered.connect(self.stop_simulation)

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

    def create_settings_button(self):
        """
        Creates the settings button and action.
        """
        self.settings_button = QtWidgets.QToolButton()
        resource_name = "gear.png"
        self.settings_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.settings_button.setText("Settings")
        self.settings_button.setStyleSheet("background-color: transparent;")
        self.settings_button.clicked.connect(self.open_settings)
