from scene import Region

from PyQt5 import QtWidgets, QtCore, QtGui


class ElectrodeGroupModel(QtCore.QAbstractItemModel):
    def __init__(self, electrode_groups, parent=None):
        super().__init__(parent)

        self._electrode_groups = electrode_groups

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return QtCore.QVariant(self._electrode_groups[index.row()].gallery)
            elif index.column() == 1:
                return QtCore.QVariant(self._electrode_groups[index.row()].wall)
            elif index.column() == 2:
                return QtCore.QVariant(self._electrode_groups[index.row()].height)

        elif role == QtCore.Qt.DecorationRole:
            if index.column() == 0:
                return QtCore.QVariant(Region.colors[index.row() % len(Region.colors)])

        return QtCore.QVariant()

    def flags(self, index):
        if not index.isValid():
            return 0

        return super().flags(index)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if parent.isValid():
            return QtCore.QModelIndex()
        else:
            return self.createIndex(row, column)

    def parent(self, index):
        return QtCore.QModelIndex()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._electrode_groups)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return 3

    def headerData(self, section, orientation, role):
        headers = ["Gallery", "Wall", "Height"]

        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole and section < len(headers):
            return QtCore.QVariant(headers[section])

        return QtCore.QVariant()


class ElectrodeGroupView(QtWidgets.QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRootIsDecorated(False)
