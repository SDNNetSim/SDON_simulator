import os
import signal
import subprocess
import sys
from PyQt5 import QtWidgets, QtCore


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Settings Menu")
        self.resize(400, 600)

        layout = QtWidgets.QVBoxLayout()

        tabs = QtWidgets.QTabWidget()

        # General Settings
        general_tab = QtWidgets.QWidget()
        general_settings_layout = QtWidgets.QFormLayout()

        self.sim_type = QtWidgets.QComboBox()
        self.sim_type.addItems(["yue", "arash"])
        self.sim_type.setToolTip(
            """
            Simulation assumptions for calculating\
            the Erlang and optical reach
            """
        )
        general_settings_layout.addRow("Sim Type:", self.sim_type)

        self.holding_time = QtWidgets.QDoubleSpinBox()
        self.holding_time.setMinimum(0.0)
        self.holding_time.setSingleStep(0.1)
        self.holding_time.setStepType(
            QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
        )
        self.holding_time.setValue(0.2)
        self.holding_time.setToolTip(
            "Mean holding time for request generation"
        )
        general_settings_layout.addRow("Holding Time:", self.holding_time)

        self.arrival_rate_start = QtWidgets.QSpinBox()
        self.arrival_rate_start.setValue(2)
        self.arrival_rate_stop = QtWidgets.QSpinBox()
        self.arrival_rate_stop.setValue(4)
        self.arrival_rate_step = QtWidgets.QSpinBox()
        self.arrival_rate_step.setValue(2)
        arrival_rate_layout = QtWidgets.QHBoxLayout()
        arrival_rate_layout.addWidget(QtWidgets.QLabel("Start:"))
        arrival_rate_layout.addWidget(self.arrival_rate_start)
        arrival_rate_layout.addWidget(QtWidgets.QLabel("Stop:"))
        arrival_rate_layout.addWidget(self.arrival_rate_stop)
        arrival_rate_layout.addWidget(QtWidgets.QLabel("Step:"))
        arrival_rate_layout.addWidget(self.arrival_rate_step)
        general_settings_layout.addRow("Arrival Rate:", arrival_rate_layout)

        self.thread_erlangs = QtWidgets.QCheckBox()
        self.thread_erlangs.setChecked(False)
        general_settings_layout.addRow("Thread Erlangs:", self.thread_erlangs)

        self.guard_slots = QtWidgets.QSpinBox()
        self.guard_slots.setMinimum(1)
        self.guard_slots.setValue(1)
        general_settings_layout.addRow("Guard Slots:", self.guard_slots)

        self.num_requests = QtWidgets.QSpinBox()
        self.num_requests.setMaximum(100000)
        self.num_requests.setValue(10000)
        general_settings_layout.addRow("Number of Requests:", self.num_requests)

        self.request_distribution = QtWidgets.QLineEdit(
            "{\"25\": 0.0, \"50\": 0.3, \"100\": 0.5, \"200\": 0.0, \"400\": 0.2}")
        general_settings_layout.addRow("Request Distribution:",
                                       self.request_distribution)

        self.max_iters = QtWidgets.QSpinBox()
        self.max_iters.setMinimum(1)
        self.max_iters.setValue(10)
        self.max_iters.setToolTip("Maximum iterations to run")
        general_settings_layout.addRow("Max Iters:", self.max_iters)

        self.max_segments = QtWidgets.QSpinBox()
        self.max_segments.setMinimum(1)
        self.max_segments.setValue(1)
        self.max_segments.setToolTip("Maximum segments for a single request")
        general_settings_layout.addRow("Max Segments:", self.max_segments)

        self.dynamic_lps = QtWidgets.QCheckBox()
        self.dynamic_lps.setChecked(False)
        self.dynamic_lps.setToolTip(
            "Use dynamic light path/segment slicing or not"
        )
        general_settings_layout.addRow("Dynamic LPS:", self.dynamic_lps)

        self.allocation_method = QtWidgets.QComboBox()
        self.allocation_method.addItems(
            ["best_fit", "first_fit", "last_fit", "priority_first",
             "priority_last", "xt_aware"]
        )
        self.allocation_method.setToolTip(
            "Method for assigning a request to a spectrum"
        )
        general_settings_layout.addRow("Allocation Method:",
                                       self.allocation_method)

        self.k_paths = QtWidgets.QSpinBox()
        self.k_paths.setValue(1)
        general_settings_layout.addRow("K Paths:", self.k_paths)

        self.route_method = QtWidgets.QLineEdit("shortest_path")
        self.route_method.setToolTip("Method for routing a request")
        general_settings_layout.addRow("Route Method:", self.route_method)

        self.save_snapshots = QtWidgets.QCheckBox()
        self.save_snapshots.setChecked(False)
        self.save_snapshots.setToolTip(
            "To save information at certain request intervals"
        )
        general_settings_layout.addRow("Save Snapshots:", self.save_snapshots)

        self.snapshot_step = QtWidgets.QSpinBox()
        self.snapshot_step.setMinimum(1)
        self.snapshot_step.setValue(10)
        self.snapshot_step.setToolTip(
            "Interval for saving snapshot results"
        )
        general_settings_layout.addRow("Snapshot Step:", self.snapshot_step)

        self.print_step = QtWidgets.QSpinBox()
        self.print_step.setValue(1)
        self.print_step.setToolTip(
            "Interval for printing simulator information"
        )
        general_settings_layout.addRow("Print Step:", self.print_step)

        general_tab.setLayout(general_settings_layout)
        tabs.addTab(general_tab, "General Settings")

        # Topology Settings
        topology_tab = QtWidgets.QWidget()
        topology_settings_layout = QtWidgets.QFormLayout()

        self.network = QtWidgets.QLineEdit("USNet")
        topology_settings_layout.addRow("Network:", self.network)

        self.spectral_slots = QtWidgets.QSpinBox()
        self.spectral_slots.setValue(128)
        topology_settings_layout.addRow("Spectral Slots:", self.spectral_slots)

        self.bw_per_slot = QtWidgets.QDoubleSpinBox()
        self.bw_per_slot.setValue(12.5)
        topology_settings_layout.addRow("BW per Slot:", self.bw_per_slot)

        self.cores_per_link = QtWidgets.QSpinBox()
        self.cores_per_link.setValue(1)
        topology_settings_layout.addRow("Cores per Link:", self.cores_per_link)

        self.const_link_weight = QtWidgets.QCheckBox()
        self.const_link_weight.setChecked(False)
        topology_settings_layout.addRow("Const Link Weight:",
                                        self.const_link_weight)

        topology_tab.setLayout(topology_settings_layout)
        tabs.addTab(topology_tab, "Topology Settings")

        # SNR Settings
        snr_tab = QtWidgets.QWidget()
        snr_settings_layout = QtWidgets.QFormLayout()

        self.snr_type = QtWidgets.QLineEdit("None")
        snr_settings_layout.addRow("SNR Type:", self.snr_type)

        self.xt_type = QtWidgets.QLineEdit("without_length")
        snr_settings_layout.addRow("XT Type:", self.xt_type)

        self.beta = QtWidgets.QDoubleSpinBox()
        self.beta.setValue(0.5)
        snr_settings_layout.addRow("Beta:", self.beta)

        self.input_power = QtWidgets.QDoubleSpinBox()
        self.input_power.setValue(0.001)
        snr_settings_layout.addRow("Input Power:", self.input_power)

        self.egn_model = QtWidgets.QCheckBox()
        self.egn_model.setChecked(False)
        snr_settings_layout.addRow("EGN Model:", self.egn_model)

        self.phi = QtWidgets.QLineEdit(
            "{\"QPSK\": 1, \"16-QAM\": 0.68, \"64-QAM\": 0.6190476190476191}")
        snr_settings_layout.addRow("Phi:", self.phi)

        self.bi_directional = QtWidgets.QCheckBox()
        self.bi_directional.setChecked(True)
        snr_settings_layout.addRow("Bi-Directional:", self.bi_directional)

        self.xt_noise = QtWidgets.QCheckBox()
        self.xt_noise.setChecked(False)
        snr_settings_layout.addRow("XT Noise:", self.xt_noise)

        self.requested_xt = QtWidgets.QLineEdit(
            "{\"QPSK\": -18.5, \"16-QAM\": -25.0, \"64-QAM\": -34.0}")
        snr_settings_layout.addRow("Requested XT:", self.requested_xt)

        snr_tab.setLayout(snr_settings_layout)
        tabs.addTab(snr_tab, "SNR Settings")

        # AI Settings
        ai_tab = QtWidgets.QWidget()
        ai_settings_layout = QtWidgets.QFormLayout()

        self.ai_algorithm = QtWidgets.QLineEdit("None")
        ai_settings_layout.addRow("AI Algorithm:", self.ai_algorithm)

        self.learn_rate = QtWidgets.QDoubleSpinBox()
        self.learn_rate.setValue(0.1)
        ai_settings_layout.addRow("Learn Rate:", self.learn_rate)

        self.discount_factor = QtWidgets.QDoubleSpinBox()
        self.discount_factor.setValue(0.9)
        ai_settings_layout.addRow("Discount Factor:", self.discount_factor)

        self.epsilon_start = QtWidgets.QDoubleSpinBox()
        self.epsilon_start.setValue(0.1)
        ai_settings_layout.addRow("Epsilon Start:", self.epsilon_start)

        self.epsilon_end = QtWidgets.QDoubleSpinBox()
        self.epsilon_end.setValue(0.01)
        ai_settings_layout.addRow("Epsilon End:", self.epsilon_end)

        self.is_training = QtWidgets.QCheckBox()
        self.is_training.setChecked(True)
        ai_settings_layout.addRow("Is Training:", self.is_training)

        ai_tab.setLayout(ai_settings_layout)
        tabs.addTab(ai_tab, "AI Settings")

        # File Settings
        file_tab = QtWidgets.QWidget()
        file_settings_layout = QtWidgets.QFormLayout()

        self.file_type = QtWidgets.QLineEdit("json")
        file_settings_layout.addRow("File Type:", self.file_type)

        file_tab.setLayout(file_settings_layout)
        tabs.addTab(file_tab, "File Settings")

        layout.addWidget(tabs)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_settings(self):
        return {
            "s1": {
                "general_settings": {
                    "sim_type": self.sim_type.currentText(),
                    "holding_time": self.holding_time.value(),
                    "arrival_rate": {
                        "start": self.arrival_rate_start.value(),
                        "stop": self.arrival_rate_stop.value(),
                        "step": self.arrival_rate_step.value()
                    },
                    "thread_erlangs": self.thread_erlangs.isChecked(),
                    "guard_slots": self.guard_slots.value(),
                    "num_requests": self.num_requests.value(),
                    "request_distribution": self.request_distribution.text(),
                    "max_iters": self.max_iters.value(),
                    "max_segments": self.max_segments.value(),
                    "dynamic_lps": self.dynamic_lps.isChecked(),
                    "allocation_method": self.allocation_method.text(),
                    "k_paths": self.k_paths.value(),
                    "route_method": self.route_method.text(),
                    "save_snapshots": self.save_snapshots.isChecked(),
                    "snapshot_step": self.snapshot_step.value(),
                    "print_step": self.print_step.value()
                },
                "topology_settings": {
                    "network": self.network.text(),
                    "spectral_slots": self.spectral_slots.value(),
                    "bw_per_slot": self.bw_per_slot.value(),
                    "cores_per_link": self.cores_per_link.value(),
                    "const_link_weight": self.const_link_weight.isChecked()
                },
                "snr_settings": {
                    "snr_type": self.snr_type.text(),
                    "xt_type": self.xt_type.text(),
                    "beta": self.beta.value(),
                    "input_power": self.input_power.value(),
                    "egn_model": self.egn_model.isChecked(),
                    "phi": self.phi.text(),
                    "bi_directional": self.bi_directional.isChecked(),
                    "xt_noise": self.xt_noise.isChecked(),
                    "requested_xt": self.requested_xt.text()
                },
                "ai_settings": {
                    "ai_algorithm": self.ai_algorithm.text(),
                    "learn_rate": self.learn_rate.value(),
                    "discount_factor": self.discount_factor.value(),
                    "epsilon_start": self.epsilon_start.value(),
                    "epsilon_end": self.epsilon_end.value(),
                    "is_training": self.is_training.isChecked()
                },
                "file_settings": {
                    "file_type": self.file_type.text()
                }
            }
        }


class SimulationThread(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    finished_signal = QtCore.pyqtSignal(str)
    output_hints_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(SimulationThread, self).__init__()
        self.simulation_process = None
        self.paused = False
        self.stopped = False
        self.mutex = QtCore.QMutex()
        self.pause_condition = QtCore.QWaitCondition()

    def run(self):
        """
        Overrides run method in QtCore.QThread
        Starting point of thread
        """
        command = os.path.join(os.getcwd(), "run_sim.py")

        self.simulation_process = subprocess.Popen(
            args=[sys.executable, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # can update progress bar with number of max iterations
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

        # Notify that simulation is finished
        self.finished_signal.emit(f'Simulation done')
        self.output_hints_signal.emit(
            f'Done...cleaning up simulation from thread'
        )

    def handle_process_state(self, process_state):
        if process_state == QtCore.QProcess.ProcessState.Starting:
            self.output_hints_signal.emit(
                f'Starting process'
            )
        elif process_state == QtCore.QProcess.ProcessState.Running:
            self.output_hints_signal.emit(
                f'Running process'
            )

    def pause(self):
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGSTOP)
            self.paused = True
            self.output_hints_signal.emit(f'Pausing simulation from thread')

    def resume(self):
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGCONT)
            self.paused = False
            self.output_hints_signal.emit(f'Resuming simulation from thread')
        self.pause_condition.wakeOne()  # Resume the thread

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGKILL)
            self.stopped = True
            self.paused = False
            self.output_hints_signal.emit(f'Stopping simulation from thread')
        # Ensure the thread exits if it was paused
        self.pause_condition.wakeOne()
