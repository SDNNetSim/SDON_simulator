import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QToolTip
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen
from PyQt5.QtCore import Qt, QPoint


class Link:
    def __init__(self, src_id, dest_id, link_len, link_id):
        self.link_id = link_id
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
        qp = QPainter(self)
        qp.begin(self)
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        qp.setBrush(QBrush(QColor(100, 150, 255), Qt.SolidPattern))
        self.draw_nodes(qp)
        qp.end()

    def draw_nodes(self, qp):
        radius = 20
        padding = 50
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        for idx, node in enumerate(self.nodes.values()):
            x = padding + (idx * (radius * 2 + 20))
            y = padding
            # qp.setBrush(QBrush(QColor(100, 150, 255), Qt.SolidPattern))
            qp.drawEllipse(QPoint(x, y), radius, radius)
            # Store position and radius for click detection
            node.position = (x, y, radius)
            # Tooltip setup
            node.tooltip = f"Link ID: {node.link_id}\nSource ID: {node.source_id}\n\
            Destination ID: {node.destination_id}"

    def mouse_press_event(self, event):
        x_click, y_click = event.x(), event.y()
        for node in self.nodes.values():
            x, y, radius = node.position
            if (x - x_click) ** 2 + (y - y_click) ** 2 <= radius ** 2:
                QToolTip.showText(event.globalPos(), node.tooltip, self)
                break


def load_nodes_from_file(filename):
    links_dict = {}
    with open(filename, 'r') as file:
        for link_id, line in enumerate(file):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                links_dict[link_id] = Link(src_id=int(parts[0]), dest_id=int(parts[1]), link_len=int(parts[2]),
                                           link_id=link_id)

    # TODO: If node already displayed, don't display again just add another link
    return links_dict


def main():
    app = QApplication(sys.argv)
    # nodes = load_nodes_from_file('nodes.txt')  # Ensure 'nodes.txt' is in the same directory
    links_dict = load_nodes_from_file('../../data/raw/us_network.txt')  # Ensure 'nodes.txt' is in the same directory
    node_display_obj = NodeDisplayWidget(links_dict)
    node_display_obj.paint_event()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()