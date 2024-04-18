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

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.progressBar = QProgressBar()
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
		self.addCentralDataDisplay()
		self.addMenuBar() # this adds the menubar
		self.addControlToolBar()
		self.initStatusBar()
	
	def addMenuBar(self):
		# Create the menu bar
		menuBar = self.menuBar()
		
		# Create File menu and add actions
		fileMenu = menuBar.addMenu('&File')
		openAction = QAction('&Load Configuration from File', self)
		openAction.triggered.connect(self.openFile)
		fileMenu.addAction(openAction)
		
		saveAction = QAction('&Save', self)
		saveAction.triggered.connect(self.saveFile)
		fileMenu.addAction(saveAction)
		
		exitAction = QAction('&Exit', self)
		exitAction.triggered.connect(self.close)
		fileMenu.addAction(exitAction)
		
		# Create Edit menu and add actions
		editMenu = menuBar.addMenu('&Edit')
		settingsAction = QAction('&Settings', self)
		settingsAction.triggered.connect(self.openSettings)
		editMenu.addAction(settingsAction)
		
		# Create Help menu and add actions
		helpMenu = menuBar.addMenu('&Help')
		aboutAction = QAction('&About', self)
		aboutAction.triggered.connect(self.about)
		helpMenu.addAction(aboutAction)
		
	def addCentralDataDisplay(self):
		# Main container widget
		containerWidget = QWidget()
		containerWidget.setStyleSheet("background-color: #a3e1a4;")  # Set the color of the main container
		
		# Layout for the container widget, allowing for margins around the central data display
		containerLayout = QVBoxLayout(containerWidget)
		containerLayout.setContentsMargins(10, 10, 10, 10)  # Adjust these margins to control the offset

		# The actual central data display widget with a white background
		dataDisplayWidget = QWidget()
		dataDisplayWidget.setStyleSheet(
			"background-color: white;"
			""
		)

		self.hoverLabel = HoverLabel()
		# Assuming this is within your main window or a relevant container/widget
		self.hoverLabel.normalText = ""
		self.hoverLabel.hoverText = ""
		self.hoverLabel.setIcon("/Users/kwadwoabempah/Python/GUI/pngtree-call-center-operator-with-phone-headset-icon-png-image_2059023.jpg")


		operator_status_info = {"ID":"n/a", "Region":"North America", "Connection Status":"Unknown"}
		
		# Connect the hoverChanged signal to a custom slot if you want to handle hover changes
		self.hoverLabel.hoverChanged.connect(lambda hovered: self.onHoverChange(self.hoverLabel, operator_status_info, hovered))
		
		containerLayout.addWidget(dataDisplayWidget)
		
		# Setting the container widget as the central widget of the main window
		self.setCentralWidget(containerWidget)

		# Example content for the data display widget
		dataLayout = QVBoxLayout(dataDisplayWidget)
		dataLabel = QLabel("Application Data Display", dataDisplayWidget)
		dataLabel.setAlignment(Qt.AlignCenter)
		# Add hoverLabel to the layout
		dataLayout.addWidget(self.hoverLabel)
		dataLayout.addWidget(dataLabel)
		
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

	def setUpSimulationThread(self):
		self.progressBar.setMaximum(1000)
		self.progressBar.setValue(0)
		self.progressBar.setVisible(True)

		self.simulation_thread = SimulationThread()
		self.simulation_thread.progressChanged.connect(self.update_progress)
		self.simulation_thread.finished.connect(self.simulation_finished)
		self.simulation_thread.start()
	
	def startSimulation(self):
		if self.start_button.text() == "Resume":
			#print("Resuming simulation")
			self.simulation_thread.resume()
			self.start_button.setText("Start")
		else:
			#print("Starting simulation")
			if not self.simulation_thread or not self.simulation_thread.isRunning():
				self.setUpSimulationThread()
			else:
				self.simulation_thread.resume()
			self.start_button.setText("Start")

	def pauseSimulation(self):
		#print("Simulation paused")
		if self.simulation_thread and self.simulation_thread.isRunning():
			self.simulation_thread.pause()
			self.start_button.setText("Resume")
	
	def resume(self):
		if self.simulation_thread and self.simulation_thread.isRunning():
			self.simulation_thread.pause()
			self.start_button.setText("Resume")  # Change button text to "Resume"
		else:
			with QMutexLocker(self.mutex):
				self.paused = False
			self.wait_cond.wakeAll()

	def stopSimulation(self):
		#print("Simulation stopped")
		if self.simulation_thread and self.simulation_thread.isRunning():
			self.simulation_thread.stop()
			self.progressBar.setValue(0)
			self.progressBar.setVisible(False)
		self.start_button.setText("Start")
	
	# Placeholder methods for menu actions
	def openFile(self):
		# Set the file dialog to filter for .yml and .json files
		file_name, _ = QFileDialog.getOpenFileName(self, "Open Configuration File", "",
												"Config Files (*.yml *.json)")
		if file_name:
			print(f"Selected file: {file_name}")
			# Here, you can add code to handle the opening and reading of the selected file

	def saveFile(self):
		print("Save file action triggered")

	def about(self):
		print("Show about dialog")
	
	def openSettings(self):
		print("Opening settings")

	def update_progress(self, value):
		self.progressBar.setValue(value)

	def simulation_finished(self):
		self.progressBar.setVisible(False)
		self.progressBar.setValue(0)
	
	def onHoverChange(self, label, data, hovered):
		if hovered:
			detailedData = "<br>".join(f"{k}: {v}" for k, v in data.items())
			tooltipText = f"Details:<br>{detailedData}"
			#print(f"Setting tooltip: {tooltipText}")  # Debug print
			label.setToolTip(tooltipText)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	

	window = MainWindow()
	window.show()
	sys.exit(app.exec_())