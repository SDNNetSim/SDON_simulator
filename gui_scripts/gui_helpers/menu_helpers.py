# pylint: disable=c-extension-no-member

from PyQt5 import QtWidgets, QtCore


class MenuHelpers(QtCore.QObject):
    """
    Contains methods related to setting up the menu and their potential options.
    """
    config_file_path_sig = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.menu_bar_obj = None  # Updated from run_gui.py script
        self.file_menu_obj = None
        self.help_menu_obj = None
        self.edit_menu_obj = None

    @staticmethod
    def load_config_file():
        """
        Loads a configuration file. Currently supported config file format is INI.
        If a config file is successfully read, it is copied to ini/run_ini/config.ini
        since all simulation scripts point to this location for configuration setup. If
        the file already exists, it is overwritten with the idea that the new file may
        contain more up-to-date configuration options. The config file path may be
        updated in the future to point to the ~/.config/sdon/config.ini

        :param:
        :return None:
        """
        config_file_dialog = QtWidgets.QFileDialog()
        file_path, _ = config_file_dialog.getOpenFileName(
            caption="Load Configuration File",
            filter="INI Files (*.ini)"
        )
        if file_path:
            target_dir = QtCore.QDir("ini/run_ini")
            target_path = target_dir.filePath("config.ini")
            if not target_dir.exists():
                target_dir.mkpath(".")
            QtCore.QFile.copy(file_path, target_path)

    # TODO: Add to standards and guidelines, must be called "create", if action must end in "action"
    def create_file_menu(self):
        """
        Creates the basis of the file menu along with adding an open action.
        """
        self.file_menu_obj = self.menu_bar_obj.addMenu('&File')
        load_config_action = QtWidgets.QAction('&Load Configuration from File', self.menu_bar_obj)
        load_config_action.triggered.connect(self.load_config_file)
        self.file_menu_obj.addAction(load_config_action)

    def create_edit_menu(self):
        """
        Creates the edit menu section.
        """
        self.edit_menu_obj = self.menu_bar_obj.addMenu('&Edit')

    def create_help_menu(self):
        """
        Creates the help menu section.
        """
        self.help_menu_obj = self.menu_bar_obj.addMenu('&Help')
