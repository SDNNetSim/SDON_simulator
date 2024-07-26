from PyQt5.QtWidgets import (
    QLabel, QSizePolicy,
)

from PyQt5.QtGui import (
    QIcon,
)

from PyQt5.QtCore import (
    QSize, pyqtSignal,
)


class HoverLabel(QLabel):
    # Optional: Signal to emit when hover changes
    hoverChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(HoverLabel, self).__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking to receive hover events
        # Set the tooltip text
        self.setToolTip("This is the data displayed on hover.")
        # Adjust the size policy to accommodate icon changes
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumSize(60, 60)  # Adjust as needed

    def setIcon(self, iconPath, size=QSize(64, 64)):
        self.icon = QIcon(iconPath)
        self.pixmap = self.icon.pixmap(size)
        self.setPixmap(self.pixmap)
        # Adjust the label size to fit the new pixmap size, if desired
        self.resize(self.pixmap.size())

    def enterEvent(self, event):
        self.setText(self.hoverText)
        self.hoverChanged.emit(True)
        super(HoverLabel, self).enterEvent(event)

    def leaveEvent(self, event):
        self.setText(self.normalText)
        self.hoverChanged.emit(False)
        super(HoverLabel, self).leaveEvent(event)

    def updateTooltip(self, newData):
        tooltipText = f"Dynamic Data: {newData}"
        self.setToolTip(tooltipText)
