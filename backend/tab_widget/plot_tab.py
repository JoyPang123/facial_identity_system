import numpy as np
import torch
from PyQt5 import QtWidgets
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolBar
from matplotlib.figure import Figure

# Using qt5 as backend
from model.triplet.model import DimFeatures

matplotlib.use("Qt5Agg")


class Canvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        plt.rcParams["axes.unicode_minus"] = False

        self.fig = Figure(dpi=dpi)
        self.fig.tight_layout()
        self.axes = self.fig.add_subplot(111)

        # Turn off tick labels and tick marks
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Policy for the canvas shape
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data, state, label, count):
        for idx in range(count, len(data)):
            self.axes.scatter(data[idx][0], data[idx][1], s=150, c="green" if state[idx - count][0] else "red",
                              zorder=10)
            self.axes.annotate(str(idx - count + 1), (data[idx][0], data[idx][1]),
                               ha="center", va="center", color="white",
                               zorder=15)

            if state[idx - count][0]:
                self.axes.plot(
                    (data[idx][0], data[state[idx - count][1]][0]), (data[idx][1], data[state[idx - count][1]][1]),
                    c="black", zorder=1, alpha=0.5
                )

        for idx in range(count):
            self.axes.scatter(data[idx][0], data[idx][1], label=label[idx], color=np.random.rand(3), zorder=10)

        self.axes.legend()

        # Turn off tick labels and tick marks
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])

        self.axes.grid(False)

    def update_figure(self, data, state, label, count):
        self.axes.cla()
        self.plot(data, state, label, count)
        self.draw()


class MatplotlibTab(QtWidgets.QWidget):
    def __init__(self, parent, bucket, database, user_id):
        super(MatplotlibTab, self).__init__(parent)

        # Set up database
        self.bucket = bucket
        self.database = database
        self.user_id = user_id

        main_layout = QtWidgets.QVBoxLayout()

        # Add the plot
        self.info_plot = Canvas(self)

        # Add the toolbar
        self.tool_bar = NavigationToolBar(self.info_plot, self)

        # Add reload button
        self.reload_btn = QtWidgets.QPushButton()
        self.reload_btn.clicked.connect(self.reload_spatial_info)
        self.reload_btn.setText("reload")
        self.reload_btn.setStyleSheet(
            """
            QPushButton {
                padding: 5px;
                border: 1px solid gray;
                border-radius: 5px;
                background-color: white;
            }
            QPushButton::hover {
                background-color: black;
                color: white;
            }
            """
        )

        bottom_h_box = QtWidgets.QHBoxLayout()
        bottom_h_box.addWidget(self.tool_bar, 1)
        bottom_h_box.addWidget(self.reload_btn)

        main_layout.addWidget(self.info_plot)
        main_layout.addLayout(bottom_h_box)

        self.setLayout(main_layout)

        # Set up model
        self.features_transform = DimFeatures().eval()
        self.features_transform.load_state_dict(torch.load("weight/plot.pt", map_location="cpu"))

    def reload_spatial_info(self):
        data = self.database.get(f"/users/{self.user_id}", "identity")
        secure_data = self.database.get(f"/users/{self.user_id}", "secure")

        if data is None or len(data) < 2:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Number of faces should be greater or equal than 2.")
            msg.setWindowTitle("Error for generating plot")
            msg.exec_()
        elif secure_data is None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("The recorded data is empty. Please try after a detection.")
            msg.setWindowTitle("Error for generating plot")
            msg.exec_()
        elif data is not None:
            faces_spatial = []
            label = []
            state = []
            name_to_idx = dict()
            count_faces = 0
            for key, value in data.items():
                count_faces += 1
                label.append(key)
                name_to_idx[key] = count_faces - 1
                faces_spatial.append(torch.tensor(value))

            for key, value in secure_data.items():
                if value["pass"]:
                    state.append((value["pass"], name_to_idx[value["name"]]))
                else:
                    state.append((value["pass"], None))
                faces_spatial.append(torch.tensor(value["features"]))

            low_dim_data = self.features_transform(torch.stack(faces_spatial)).tolist()
            self.info_plot.update_figure(low_dim_data, state, label, count_faces)
