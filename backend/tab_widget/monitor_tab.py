from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QRegExpValidator
from PyQt5.QtCore import Qt, QTimer, QEventLoop, QRegExp

import numpy as np

from backend.tab_widget.video_thread import VideoWorkerThread


class MonitorTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super(MonitorTab, self).__init__(parent)

        self.video_display_label = QtWidgets.QLabel()
        self.video_display_label.setFixedSize(500, 500)

        # Set up line edit validator for input ip-address
        self.ip_address = QtWidgets.QLineEdit()
        self.ip_address.setPlaceholderText("IP Address")
        self.ip_address.setInputMask("000.000.000.000")

        self.start_button = QtWidgets.QPushButton("Start Monitor")
        self.start_button.clicked.connect(self.start_video)
        self.start_button.setStyleSheet(
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

        self.stop_button = QtWidgets.QPushButton("Stop Monitor")
        self.stop_button.clicked.connect(self.stop_current_video)
        self.stop_button.setStyleSheet(
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

        side_panel_v_box = QtWidgets.QVBoxLayout()
        side_panel_v_box.setAlignment(Qt.AlignTop)
        side_panel_v_box.addWidget(self.ip_address)
        side_panel_v_box.addWidget(self.start_button)
        side_panel_v_box.addWidget(self.stop_button)

        side_panel_frame = QtWidgets.QFrame()
        side_panel_frame.setMinimumWidth(150)
        side_panel_frame.setLayout(side_panel_v_box)

        main_h_box = QtWidgets.QHBoxLayout()
        main_h_box.addWidget(self.video_display_label, 1)
        main_h_box.addWidget(side_panel_frame)

        self.setLayout(main_h_box)

        self.thread_is_running = False

    def start_video(self):
        self.thread_is_running = True
        self.start_button.setEnabled(False)
        self.start_button.repaint()

        video_file = f"udp:{self.ip_address.text()}"
        self.video_thread_worker = VideoWorkerThread(self, video_file)

        self.video_thread_worker.frame_data_updated.connect(self.update_video_frames)
        self.video_thread_worker.start()

    def stop_current_video(self):
        if self.thread_is_running:
            self.thread_is_running = False
            self.video_thread_worker.stop_thread()

            self.video_display_label.clear()
            self.start_button.setEnabled(True)

    def update_video_frames(self, video_frame):
        height, width, channels = video_frame.shape
        bytes_per_line = width * channels

        converted_Qt_image = QImage(video_frame.copy(), width, height, bytes_per_line, QImage.Format_RGB888)

        self.video_display_label.setPixmap(
            QPixmap.fromImage(converted_Qt_image).scaled(
                self.video_display_label.width(), self.video_display_label.height())
        )

    @staticmethod
    def pix_to_array(pixmap):
        h = pixmap.size().height()
        w = pixmap.size().width()

        q_image = pixmap.toImage()
        byte_str = q_image.bits().asstring(w * h * 4)

        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((h, w, 4))
        return img
