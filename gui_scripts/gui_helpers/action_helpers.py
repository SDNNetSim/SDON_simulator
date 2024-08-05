# pylint: disable=c-extension-no-member
import networkx as nx
from matplotlib import pyplot as plt

from PyQt5 import QtWidgets, QtCore, QtGui

from gui_scripts.gui_helpers.general_helpers import SettingsDialog
from data_scripts.structure_data import create_network


class NodeInfoDialog(QtWidgets.QDialog):
    def __init__(self, node, info, parent=None):
        super(NodeInfoDialog, self).__init__(parent)
        self.setWindowTitle(f"Node Information - {node}")
        self.setGeometry(100, 100, 300, 200)

        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(f"Node: {node}\nInfo: {info}")
        layout.addWidget(info_label)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class ActionHelpers:
    """
    Contains methods related to performing actions.
    """

    def __init__(self):
        self.menu_bar_obj = None  # Updated from run_gui.py script
        self.menu_help_obj = None  # Created in menu_helpers.py
        self.mw_topology_view_area = None  # Updated from the run_gui.py script

    @staticmethod
    def save_file():
        """
        Saves a file.
        """
        print("Save file action triggered.")

    @staticmethod
    def about():
        """
        Shows about dialog.
        """
        print("Show about dialog.")

    @staticmethod
    def open_settings():
        """
        Opens the settings panel.
        """
        settings_dialog = SettingsDialog()
        settings_dialog.setModal(True)
        settings_dialog.setStyleSheet("""
            background-color: white;
        """)
        if settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            print(settings_dialog.get_settings())

    def _display_topology(self, net_name: str):
        topology_information_dict = create_network(net_name=net_name)

        edge_list = [(src, des, {'weight': link_len})
                     for (src, des), link_len in
                     topology_information_dict.items()]
        network_topo = nx.Graph(edge_list)

        figure, axis = plt.subplots(figsize=(8, 6), dpi=100)
        pos = nx.spring_layout(network_topo, seed=5, scale=3.5)

        # Draw edges and labels using NetworkX
        nx.draw_networkx_edges(network_topo, pos, ax=axis)
        nx.draw_networkx_labels(network_topo, pos, ax=axis, font_size=8)

        # Draw nodes using scatter to enable picking
        x, y = zip(*pos.values())
        scatter = axis.scatter(x, y, s=200, picker=True, zorder=2)
        print("Nodes plotted with picking enabled")  # Debugging line

        def on_pick(event):
            print("Pick event triggered")  # Debugging line
            ind = event.ind[0]
            print(f"Node index picked: {ind}")  # Debugging line
            node = list(network_topo.nodes())[ind]
            info = "Additional Info: ..."  # Replace with actual node information
            dialog = NodeInfoDialog(node, info, self.menu_bar_obj)
            dialog.exec_()

        def on_hover(event):
            if event.inaxes == axis:
                for i, node_pos in enumerate(zip(x, y)):
                    if (event.xdata - node_pos[0]) ** 2 + (event.ydata - node_pos[1]) ** 2 < 0.05:
                        node = list(network_topo.nodes())[i]
                        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Node: {node}")
                        break  # Show tooltip for only one node at a time

        # Connect event handlers to the figure canvas
        figure.canvas.mpl_connect('pick_event', on_pick)
        figure.canvas.mpl_connect('motion_notify_event', on_hover)
        print("Event handlers connected")  # Debugging line

        # Enable interactive mode
        plt.ion()
        print("Interactive mode enabled")  # Debugging line

        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        buffer = figure.canvas.buffer_rgba()
        image = QtGui.QImage(buffer, width, height, QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(image)

        label = QtWidgets.QLabel(self.menu_bar_obj)
        label.setFixedSize(pixmap.rect().size())
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setPixmap(pixmap)

        self.mw_topology_view_area.setWidget(label)
        print("Topology displayed")  # Debugging line

    def display_topology(self):
        """
        Displays a network topology.
        """
        network_selection_dialog = QtWidgets.QDialog()
        network_selection_dialog.setSizeGripEnabled(True)

        dialog_pos = self.menu_bar_obj.mapToGlobal(
            self.menu_bar_obj.rect().center()) - network_selection_dialog.rect().center()  # Center window
        network_selection_dialog.move(dialog_pos)

        network_selection_input = QtWidgets.QInputDialog()
        # TODO: Hard coded, should read potential files we have?
        items = ['USNet', 'NSFNet', 'Pan-European']
        net_name, valid_net_name = network_selection_input.getItem(
            network_selection_dialog, "Choose a network type:",
            "Select Network Type", items, 0, False
        )

        if valid_net_name and net_name:
            self._display_topology(net_name=net_name)
        else:
            raise NotImplementedError(f"{net_name} is not a valid network name.")

    def create_topology_action(self):
        """
        Creates the action to display a topology properly.
        """
        display_topology_action = QtWidgets.QAction('&Display topology', self.menu_bar_obj)
        display_topology_action.triggered.connect(self.display_topology)
        self.menu_help_obj.file_menu_obj.addAction(display_topology_action)

    def create_save_action(self):
        """
        Create a save action to save a file.
        """
        save_action = QtWidgets.QAction('&Save', self.menu_bar_obj)
        save_action.triggered.connect(self.save_file)
        self.menu_help_obj.file_menu_obj.addAction(save_action)

    def create_exit_action(self):
        """
        Create an exit action to exit a simulation run.
        """
        exit_action = QtWidgets.QAction('&Exit', self.menu_bar_obj)
        exit_action.triggered.connect(self.menu_bar_obj.close)
        self.menu_help_obj.file_menu_obj.addAction(exit_action)

    def create_settings_action(self):
        """
        Create a settings action to trigger a display of the settings panel.
        """
        settings_action = QtWidgets.QAction('&Settings', self.menu_bar_obj)
        settings_action.triggered.connect(self.open_settings)
        self.menu_help_obj.edit_menu_obj.addAction(settings_action)

    def create_about_action(self):
        """
        Create about action to display relevant about information regarding the simulator.
        """
        about_action = QtWidgets.QAction('&About', self.menu_bar_obj)
        about_action.triggered.connect(self.about)
        self.menu_help_obj.help_menu_obj.addAction(about_action)
