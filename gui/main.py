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

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.start_button = QToolButton()
		self.pause_button = QToolButton()
		self.stop_button = QToolButton()
		self.simulation_thread = None
		self.initUI()

	def initUI(self):
		self.setWindowTitle("SDNv1")
		self.resize(1280, 720)  # Set initial size of the window
		self.setStyleSheet("background-color: #a3e1a4")  # Set light gray background color
		self.centerWindow()
		self.addControlToolBar()
		self.initStatusBar()
	
		
	def addControlToolBar(self):
		# Create toolbar and add actions
		toolbar = self.addToolBar("Simulation Controls")
		# Set gray background color and black text color for the toolbar
		toolbar.setStyleSheet("background-color: #d2dae2; color: white;")
		toolbar.setMovable(False)
		toolbar.setIconSize(QSize(20,20))

  		# Create custom tool button for Start action with transparent background
		#start_button = QToolButton()
		self.start_button.setIcon(QIcon("./media/light-green-play-button.png"))
		self.start_button.setText("Start")
		self.start_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
		self.start_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
		self.start_button.clicked.connect(self.startSimulation)

		# set up for pause button
		#pause_button = QToolButton()
		self.pause_button.setIcon(QIcon("./media/pause.png"))
		self.pause_button.setText("Pause")
		self.pause_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
		self.pause_button.setStyleSheet("background-color: transparent;")  # Set transparent background color
		self.pause_button.clicked.connect(self.pauseSimulation)

		# set up for stop button
		#stop_button = QToolButton()
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

	def initStatusBar(self):
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