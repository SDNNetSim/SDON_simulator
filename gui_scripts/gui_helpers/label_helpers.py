# pylint: disable=c-extension-no-member
# pylint: disable=no-name-in-module
# pylint: disable=super-with-arguments

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
    """
    Handles all labels for hover actions.
    """
    hover_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(HoverLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setToolTip("This is the data displayed on hover.")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumSize(60, 60)

        self.icon = None
        self.pixmap = None

    def set_icon(self, icon_path, size=QSize(64, 64)):
        """
        Sets a specific icon.
        :param icon_path: The icon location.
        :param size: Desired size of the icon.
        """
        self.icon = QIcon(icon_path)
        self.pixmap = self.icon.pixmap(size)
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.size())

    def enter_event(self, event):
        """
        Enters an event via text.

        :param event: The event to be emitted.
        """
        self.setText(self.hoverText)
        self.hover_changed.emit(True)
        super(HoverLabel, self).enterEvent(event)

    def leave_event(self, event):
        """
        Leaves an event.

        :param event: The event to remove.
        """
        self.setText(self.normalText)
        self.hover_changed.emit(False)
        super(HoverLabel, self).leaveEvent(event)

    def update_tool_tip(self, new_data):
        """
        Updates to a tool tip text.

        :param new_data: Text to update with.
        """
        tool_tip_text = f"Dynamic Data: {new_data}"
        self.setToolTip(tool_tip_text)
