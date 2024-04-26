import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QToolTip
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen
from PyQt5.QtCore import Qt, QPoint


class Node:
    def __init__(self, node_id, src_id, dest_id, link_len):
        self.node_id = node_id
        self.source_id = src_id
        self.destination_id = dest_id
        self.distance = link_len


class NodeDisplayWidget(QWidget):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.init_ui()

    def init_ui(self):
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Node Display')
        self.setToolTip('Click on a node to see more info')
        self.show()

    def paint_event(self):
        qp = QPainter()
        qp.begin(self)
        self.draw_nodes(qp)
        qp.end()

    def draw_nodes(self, qp):
        radius = 20
        padding = 50
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        for idx, node in enumerate(self.nodes.values()):
            x = padding + (idx * (radius * 2 + 20))
            y = padding
            qp.setBrush(QBrush(QColor(100, 150, 255), Qt.SolidPattern))
            qp.drawEllipse(QPoint(x, y), radius, radius)
            # Store position and radius for click detection
            node.position = (x, y, radius)
            # Tooltip setup
            node.tooltip = f"Node ID: {node.node_id}\nSource ID: {node.source_id}\n\
            Destination ID: {node.destination_id}"

    def mouse_press_event(self, event):
        x_click, y_click = event.x(), event.y()
        for node in self.nodes.values():
            x, y, radius = node.position
            if (x - x_click) ** 2 + (y - y_click) ** 2 <= radius ** 2:
                QToolTip.showText(event.globalPos(), node.tooltip, self)
                break


def load_nodes_from_file(filename):
    nodes = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                node = Node(int(parts[0]), int(parts[1]), int(parts[2]))
                nodes[node.node_id] = node
    return nodes


def main():
    app = QApplication(sys.argv)
    nodes = load_nodes_from_file('nodes.txt')  # Ensure 'nodes.txt' is in the same directory
    NodeDisplayWidget(nodes)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
