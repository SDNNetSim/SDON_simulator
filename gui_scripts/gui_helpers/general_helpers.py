# pylint: disable=c-extension-no-member

import os
import signal
import subprocess
import sys

from PyQt5 import QtWidgets, QtCore

from gui_scripts.gui_args.config_args import SETTINGS_CONFIG_DICT


class SettingsDialog(QtWidgets.QDialog):  # pylint: disable=too-few-public-methods
    """
    The settings window in the menu bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings Menu")
        self.resize(400, 600)
        self.layout = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.settings_widgets = {}
        self._setup_layout()

        self.setLayout(self.layout)

    def _setup_layout(self):
        for category in SETTINGS_CONFIG_DICT:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QFormLayout()
            for setting in category["settings"]:
                widget, label = self._create_widget(setting)
                self.settings_widgets[label] = widget
                tab_layout.addRow(label, widget)
            tab.setLayout(tab_layout)
            self.tabs.addTab(tab, category["category"])
        self.layout.addWidget(self.tabs)

        self._setup_buttons()

    @staticmethod
    def _create_widget(setting):
        widget_type = setting["type"]
        label = setting["label"]
        if widget_type == "combo":
            widget = QtWidgets.QComboBox()
            widget.addItems(setting["options"])
            widget.setCurrentText(setting["default"])
        elif widget_type == "check":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(setting["default"])
        elif widget_type == "line":
            widget = QtWidgets.QLineEdit(setting["default"])
        elif widget_type == "spin":
            widget = QtWidgets.QSpinBox()
            widget.setValue(setting["default"])
            widget.setMinimum(setting.get("min", 0))
            widget.setMaximum(setting.get("max", 100))
        elif widget_type == "double_spin":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setValue(setting["default"])
            widget.setMinimum(setting.get("min", 0.0))
            widget.setSingleStep(setting.get("step", 1.0))
        return widget, label

    def _setup_buttons(self):
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def get_settings(self):
        """
        Gets and structures all configuration settings.

        :return: The simulation configuration.
        :rtype: dict
        """
        settings = {}
        for category in SETTINGS_CONFIG_DICT:
            category_name = category["category"].lower() + "_settings"
            settings[category_name] = {}
            for setting in category["settings"]:
                label = setting["label"]
                widget = self.settings_widgets[label]
                settings[category_name][self._format_label(label)] = self._get_widget_value(widget)
        return {"s1": settings}

    @staticmethod
    def _format_label(label):
        return label.lower().replace(" ", "_").replace(":", "")

    @staticmethod
    def _get_widget_value(widget):
        resp = None
        if isinstance(widget, QtWidgets.QComboBox):
            resp = widget.currentText()
        elif isinstance(widget, QtWidgets.QCheckBox):
            resp = widget.isChecked()
        elif isinstance(widget, QtWidgets.QLineEdit):
            resp = widget.text()
        elif isinstance(widget, QtWidgets.QSpinBox):
            resp = widget.value()
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            resp = widget.value()

        return resp


class SimulationThread(QtCore.QThread):
    """
    Sets up simulation thread runs.
    """
    progress_changed = QtCore.pyqtSignal(int)
    finished_signal = QtCore.pyqtSignal(str)
    output_hints_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(SimulationThread, self).__init__()  # pylint: disable=super-with-arguments

        self.simulation_process = None
        self.paused = False
        self.stopped = False
        self.mutex = QtCore.QMutex()
        self.pause_condition = QtCore.QWaitCondition()

    def _run(self):
        for output_line in self.simulation_process.stdout:
            with QtCore.QMutexLocker(self.mutex):
                if self.stopped:
                    break

                while self.paused:
                    self.pause_condition.wait(
                        self.mutex
                    )

            self.output_hints_signal.emit(output_line)

        self.simulation_process.stdout.close()
        self.simulation_process.wait()

        self.finished_signal.emit('Simulation done')
        self.output_hints_signal.emit('Done...cleaning up simulation from thread')

    def run(self):
        """
        Overrides run method in QtCore.QThread.
        """
        command = os.path.join(os.getcwd(), "run_sim.py")

        self.simulation_process = subprocess.Popen(  # pylint: disable=consider-using-with
            args=[sys.executable, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self._run()

    def handle_process_state(self, process_state):
        """
        Starts or runs a specific process.

        :param process_state: The current state of the process.
        """
        if process_state == QtCore.QProcess.ProcessState.Starting:
            self.output_hints_signal.emit('Starting process')
        elif process_state == QtCore.QProcess.ProcessState.Running:
            self.output_hints_signal.emit('Running process')

    def pause(self):
        """
        Pauses a single simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGSTOP)
            self.paused = True
            self.output_hints_signal.emit('Pausing simulation from thread')

    def resume(self):
        """
        Resumes a simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGCONT)
            self.paused = False
            self.output_hints_signal.emit('Resuming simulation from thread')
        self.pause_condition.wakeOne()

    def stop(self):
        """
        Stops a simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGKILL)
            self.stopped = True
            self.paused = False
            self.output_hints_signal.emit('Stopping simulation from thread')
        self.pause_condition.wakeOne()


class DirectoryTreeView(QtWidgets.QTreeView):
    item_double_clicked_sig = QtCore.pyqtSignal(QtCore.QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionBehavior(QtWidgets.QTreeView.SelectRows)
        self.setSelectionMode(QtWidgets.QTreeView.SingleSelection)

    def mousePressEvent(self, event):
        """
        Overrides mousePressEvent in QTreeView for single press
        """
        index = self.indexAt(event.pos())
        if event.button() == QtCore.Qt.LeftButton and index.isValid():
            self.setCurrentIndex(index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.pos())
        if event.button() == QtCore.Qt.LeftButton and index.isValid():
            self.item_double_clicked_sig.emit(index)
        super().mouseDoubleClickEvent(event)
