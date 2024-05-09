from PyQt5 import QtCore


class SimulationThread(QtCore.QThread):
    # Signal emitted to update progress
    progress_changed_sig = QtCore.pyqtSignal(int)
    # Signal emitted when simulation finishes
    finished_sig = QtCore.pyqtSignal()

    def __init__(self):
        super(SimulationThread, self).__init__()
        self.paused = False
        self.stopped = False
        self.mutex = QtCore.QMutex()
        self.pause_condition = QtCore.QWaitCondition()

    def run(self):
        """
        Overrides run method in QtCore.QThread
        Starting point of thread
        """
        for i in range(1, 1001):
            with QtCore.QMutexLocker(self.mutex):
                if self.stopped:
                    break  # Exit loop if stopped

                while self.paused:
                    self.pause_condition.wait(
                        self.mutex)  # Wait until resume() is called

            self.progress_changed_sig.emit(i)  # Emit progress
            self.msleep(2)  # Simulate work

        self.finished_sig.emit()  # Notify that simulation is finished

    def pause(self):
        with QtCore.QMutexLocker(self.mutex):
            self.paused = True

    def resume(self):
        with QtCore.QMutexLocker(self.mutex):
            self.paused = False
        self.pause_condition.wakeOne()  # Resume the thread

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True
        # Ensure the thread exits if it was paused
        self.pause_condition.wakeOne()
