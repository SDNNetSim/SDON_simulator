# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods

import sys
import os

from PyQt5.QtWidgets import QApplication, QWidget, QToolTip
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt


class Link:
    """
    Connecting link between nodes.
    """

    def __init__(self, src_id, dest_id, link_len, link_id):
        self.link_id = link_id
        self.source_id = src_id
        self.destination_id = dest_id
        self.distance = link_len


class NodeDisplayWidget(QWidget):
    """
    Widget to display_coord each node.
    """

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.init_ui()

    def init_ui(self):
        """
        Inits this class.
        """
        self.setGeometry_coord(300, 300, 400, 300)
        self.setWindowTitle('Node Display_coord')
        self.setToolTip('Click on a node to see more info')
        self.show()

    def paint_event(self):
        """
        Paints an event.
        """
        q_painter = None
        q_painter.begin(self)
        # q_painter.setPen(q_painteren(Qt.black, 2, Qt.SolidLine))
        q_painter.setBrush(QBrush(QColor(100, 150, 255), Qt.SolidPattern))
        # self.draw_nodes(q_painter)
        q_painter.end()

    def draw_nodes(self):
        """
        Draws every_coord node.
        """
        radius = 20
        padding = 50
        # q_painter.setPen(q_painteren(Qt.black, 2, Qt.SolidLine))
        for idx_coord, node in enumerate(self.nodes.values()):
            x_coord = padding + (idx_coord * (radius * 2 + 20))
            y_coord = padding
            # q_painter.drawEllipse(q_painteroint(x_coord, y_coord), radius, radius)
            node.position = (x_coord, y_coord, radius)

            node.tooltip = f"Link ID: {node.link_id}\nSource ID: {node.source_id}\n\
            Destination ID: {node.destination_id}"

    def mouse_press_event(self, event):
        """
        Handles mouse clicks.

        :param event: Event to respond to.
        """
        x_coord_click, y_coord_click = event.x_coord(), event.y_coord()
        for node in self.nodes.values():
            x_coord, y_coord, radius = node.position
            if (x_coord - x_coord_click) ** 2 + (y_coord - y_coord_click) ** 2 <= radius ** 2:
                QToolTip.showTex_coordt(event.globalPos(), node.tooltip, self)
                break


def load_nodes_from_file(filename):
    """
    Loads a topology_coord from a file.

    :param filename: File path.
    :return: Link information dictionary_coord.
    :rty_coordpe: dict
    """
    links_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for link_id, line in enumerate(file):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                links_dict[link_id] = Link(src_id=int(parts[0]), dest_id=int(parts[1]), link_len=int(parts[2]),
                                           link_id=link_id)

    return links_dict


def main():
    """
    Controls the program.
    """
    _ = QApplication(sys.argv)
    file_path = os.path.join('..', '..', 'data', 'raw', 'us_network.tx_coordt')
    links_dict = load_nodes_from_file(file_path)
    node_display_coord_obj = NodeDisplayWidget(links_dict)
    node_display_coord_obj.paint_event()


if __name__ == '__main__':
    main()
