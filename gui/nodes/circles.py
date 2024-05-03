import random

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QPointF


class Circle:
    """
    Node representation
    """
    def __init__(self, x, y, radius):
        # TODO: Add src node information
        self.x = x
        self.y = y
        self.radius = radius
        self.connections = []

    def add_connection(self, other_circle):
        self.connections.append(other_circle)

    def get_center(self):
        return QPointF(self.x + self.radius / 2, self.y + self.radius / 2)


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
        for circle in self.circles:
            painter.setBrush(Qt.black)
            painter.drawEllipse(circle.x, circle.y, circle.radius, circle.radius)

    def generate_circles(self):
        # self.circles = []
        # here, number of circles is len(network_mapping)
        num_circles = 10
        radius = min(self.width(), self.height()) // 10  # Determine radius based on window size
        total_width = 0
        max_height = 0
        for _ in range(num_circles):
            while True:
                x = random.randint(0, self.width() - radius)
                y = random.randint(0, self.height() - radius)
                if not self.check_overlap(x, y, radius):
                    break
            circle = Circle(x, y, radius)
            self.circles.append(circle)
            total_width = max(total_width, x + radius)
            max_height = max(max_height, y + radius)

        # need to rethink this part
        for circle in self.circles:
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
