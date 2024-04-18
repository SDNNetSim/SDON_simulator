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

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setWindowTitle("SDNv1")
		self.resize(1280, 720)  # Set initial size of the window
		self.setStyleSheet("background-color: #a3e1a4")  # Set light gray background color
		self.centerWindow()
	
	def centerWindow(self):
		# Calculate the center point of the screen
		center_point = QDesktopWidget().screenGeometry().center()
		# Reposition window in center of screen
		self.move(center_point - self.rect().center())

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())