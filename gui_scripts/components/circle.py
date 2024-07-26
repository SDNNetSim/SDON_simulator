import random
import math

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF


class Circle:
    """
    Node representation
    """

    def __init__(self, x, y, radius, index):
        # TODO: Add src node information
        self.x = x
        self.y = y
        self.index = index
        self.radius = radius
        self.connections = []

    def add_connection(self, other_circle):
        self.connections.append(other_circle)

    def get_center(self):
        # TODO: Fine tune it
        return QPointF(self.x + self.radius / 2.0, self.y + self.radius / 2.0)


# TODO: Change to NodeWidget
#   - Change location setting
#   - Have a data structure with coordinates for each network
#       - {'0': {'x': 100, 'y': 50}}
#       - Make it responsive
#       - On Overflow (scroll)
class CirclesWidget(QWidget):
    """
    Class to represent nodes
    """

    def __init__(self):
        super().__init__()
        # don't replace but pass network_mapping to this widget
        self.circles = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(12)  # Set the font size for the numbers
        painter.setFont(font)
        for circle in self.circles:
            painter.setBrush(Qt.black)
            painter.drawEllipse(circle.x, circle.y, circle.radius, circle.radius)

            # TODO: ChatGPT
            painter.setPen(Qt.white)
            center = circle.get_center()
            painter.drawText(center, str(self.circles.index(circle) + 1))

        # Draw connections between circles
        painter.setPen(QPen(Qt.red, 3))
        for circle in self.circles:
            for connected_circle in circle.connections:
                start_point = circle.get_center() - QPointF(19.09, 0)
                end_point = connected_circle.get_center() - QPointF(19.1, 0)
                # Draw line between adjusted points
                # TODO: May need to change per node
                painter.drawLine(start_point, end_point)

    def generate_circles(self):
        # self.circles = []
        # here, number of circles is len(network_mapping)
        num_circles = 10
        radius = min(self.width(), self.height()) // 10  # Determine radius based on window size
        total_width = 0
        max_height = 0
        for index in range(num_circles):
            while True:
                x = random.randint(0, self.width() - radius)
                y = random.randint(0, self.height() - radius)
                if not self.check_overlap(x, y, radius):
                    break
            circle = Circle(x, y, radius, index=index)
            self.circles.append(circle)
            total_width = max(total_width, x + radius)
            max_height = max(max_height, y + radius)

        # need to rethink this part
        for circle in self.circles:
            # TODO: Make sure connections from topology
            num_connections = random.randint(1, min(3, num_circles - 1))
            connected_circles = random.sample(self.circles, num_connections)
            for connected_circle in connected_circles:
                if connected_circle != circle:
                    circle.add_connection(connected_circle)

        self.update()

    def check_overlap(self, x, y, radius):
        """
        Checks overlap for
        """
        for circle in self.circles:
            dx = circle.x - x
            dy = circle.y - y
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance < radius + circle.radius:
                return True
        return False