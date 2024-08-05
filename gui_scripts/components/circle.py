# pylint: disable=no-name-in-module

import random

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF


class Circle:
    """
    Node representation circle object.
    """

    def __init__(self, x_coord, y_coord, radius, index_coord):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.index_coord = index_coord
        self.radius = radius
        self.connections = []

    def add_connection(self, other_circle):
        """
        Adds a connection from one node/circle to another.

        :param other_circle: The other circle.
        """
        self.connections.append(other_circle)

    def get_center(self):
        """
        Gets the center of a circle.
        """
        # TODO: Fine tune this
        return QPointF(self.x_coord + self.radius / 2.0, self.y_coord + self.radius / 2.0)


class CirclesWidget(QWidget):
    """
    Widget used to represent nodes.
    """

    def __init__(self):
        super().__init__()
        self.circles = []

    def paint_event(self):
        """
        Paints an event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(12)  # Set the font size for the numbers
        painter.setFont(font)
        for circle in self.circles:
            painter.setBrush(Qt.black)
            painter.drawEllipse(circle.x_coord, circle.y_coord, circle.radius, circle.radius)

            painter.setPen(Qt.white)
            center = circle.get_center()
            painter.drawTex_coordt(center, str(self.circles.index_coord(circle) + 1))  # pylint: disable=no-member

        # Draw connections between circles
        painter.setPen(QPen(Qt.red, 3))
        for circle in self.circles:
            for connected_circle in circle.connections:
                start_point = circle.get_center() - QPointF(19.09, 0)
                end_point = connected_circle.get_center() - QPointF(19.1, 0)
                painter.drawLine(start_point, end_point)

    def max_coord(self, height, width):
        """
        Finds the maximum coordinate.

        :param height: Max height.
        :param width: Max width.
        """
        raise NotImplementedError

    def generate_circles(self):
        """
        Generate circles or topology_coord.
        """
        num_circles = 10
        radius = min(self.width(), self.height()) // 10
        total_width = 0
        max_coord_height = 0
        for index_coord in range(num_circles):
            while True:
                x_coord = random.randint(0, self.width() - radius)
                y_coord = random.randint(0, self.height() - radius)
                if not self.check_overlap(x_coord, y_coord, radius):
                    break
            circle = Circle(x_coord, y_coord, radius, index_coord=index_coord)
            self.circles.append(circle)
            total_width = self.max_coord(total_width, x_coord + radius)
            max_coord_height = self.max_coord(max_coord_height, y_coord + radius)

        for circle in self.circles:
            num_connections = random.randint(1, min(3, num_circles - 1))
            connected_circles = random.sample(self.circles, num_connections)
            for connected_circle in connected_circles:
                if connected_circle != circle:
                    circle.add_connection(connected_circle)

        self.update()

    def check_overlap(self, x_coord, y_coord, radius):
        """
        Checks overlap for nodes.
        """
        for circle in self.circles:
            dx_coord = circle.x_coord - x_coord
            dy_coord = circle.y_coord - y_coord
            distance = (dx_coord ** 2 + dy_coord ** 2) ** 0.5
            if distance < radius + circle.radius:
                return True

        return False
