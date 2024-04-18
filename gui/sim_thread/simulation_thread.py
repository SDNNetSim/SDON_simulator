from PqQt5.QtCore import (
	QThread, QMutex, QWaitCondition,
	QMutexLocker, pyqtSignal, 
)

class SimulationThread(QThread):
	progressChanged = pyqtSignal(int)  # Signal emitted to update progress
	finished = pyqtSignal()  # Signal emitted when simulation finishes

	def __init__(self):
		super(SimulationThread, self).__init__()
		self.paused = False
		self.stopped = False
		self.mutex = QMutex()
		self.pause_condition = QWaitCondition()

	def run(self):
		for i in range(1, 1001):
			with QMutexLocker(self.mutex):
				if self.stopped:
					break  # Exit loop if stopped

				while self.paused:
					self.pause_condition.wait(self.mutex)  # Wait until resume() is called

			self.progressChanged.emit(i)  # Emit progress
			self.msleep(10)  # Simulate work

		self.finished.emit()  # Notify that simulation is finished
	
	def pause(self):
		with QMutexLocker(self.mutex):
			self.paused = True

	def resume(self):
		with QMutexLocker(self.mutex):
			self.paused = False
		self.pause_condition.wakeOne()  # Resume the thread

	def stop(self):
		with QMutexLocker(self.mutex):
			self.stopped = True
		self.pause_condition.wakeOne()  # Ensure the thread exits if it was paused