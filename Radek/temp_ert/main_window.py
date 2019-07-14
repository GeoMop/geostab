from scene import DiagramView
import electrode_parser
from electrode_views import ElectrodeGroupModel, ElectrodeGroupView
from measurement_view import MeasurementModel, MeasurementView
#import ert_prepare
#from run_inv import RunInvDlg

from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("geostab")

        self.resize(1200, 800)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        hlayout = QtWidgets.QHBoxLayout(central_widget)
        vlayout = QtWidgets.QVBoxLayout()
        hlayout.addLayout(vlayout)

        file_name = "seznam souřadnic ERT bukov_finale_pb 4.xlsx"
        res = electrode_parser.parse(file_name)
        self._electrode_groups = res["electrode_groups"]
        self._measurements = res["measurements"]

        self._electrode_group_model = ElectrodeGroupModel(self._electrode_groups)

        self.el_groupView = ElectrodeGroupView(central_widget)
        self.el_groupView.setModel(self._electrode_group_model)
        vlayout.addWidget(self.el_groupView)
        self.el_groupView.setMinimumWidth(200)
        self.el_groupView.setMaximumWidth(400)

        self._measurement_model = MeasurementModel(self._measurements)

        self.measurementView = MeasurementView(central_widget)
        self.measurementView.setModel(self._measurement_model)
        vlayout.addWidget(self.measurementView)
        self.measurementView.setMinimumWidth(200)
        self.measurementView.setMaximumWidth(400)

        self.run_invButton = QtWidgets.QPushButton("Run inversion", central_widget)
        self.run_invButton.clicked.connect(self._handle_run_intButton)
        vlayout.addWidget(self.run_invButton)

        self.diagram_view = DiagramView()
        hlayout.addWidget(self.diagram_view)

        self.diagram_view.show_electrodes(self._electrode_groups)
        self.diagram_view.show_map()
        self.diagram_view.fitInView(self.diagram_view._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    # def _handle_el_groupList_item_change(self):
    #     currentItem = self.el_groupView.currentItem()
    #     if currentItem:
    #         self.show_electrodes(currentItem.text())

    # def load_electrodes(self):
    #     electrode_list = electrode_parser.parse("seznam souřadnic ERT bukov_finale_ff.xlsx")
    #     return
    #     self.electrode_dict = {e.id: e for e in electrode_list}
    #
    #     self.electrode_group_dict = {}
    #     for k, v in self.electrode_dict.items():
    #         gk = "{}, {}, {}".format(v.dilo, v.wall.name, v.height)
    #         if gk in self.electrode_group_dict:
    #             self.electrode_group_dict[gk].append(v)
    #         else:
    #             self.electrode_group_dict[gk] = [v]

    # def show_electrode_groups(self):
    #     self.el_groupView.clear()
    #     self.el_groupView.addItems(sorted(self.electrode_group_dict.keys()))

    # def show_electrodes(self, key):
    #     self.electrodeList.clear()
    #     self.electrodeList.addItems(sorted(["{}, {}, {}, {}, {}".format(e.id, e.metraz, e.x, e.y, e.z) for e in self.electrode_group_dict[key]]))

    def _handle_run_intButton(self):
        measurements = self._measurement_model.checkedMeasurements()
        if measurements:
            print(measurements)
            #data = ert_prepare.prepare(self._electrode_groups, measurements)
            #data.save("out.dat")
            from run_inv import RunInvDlg
            dlg = RunInvDlg(self._electrode_groups, measurements, self)
            dlg.exec()
