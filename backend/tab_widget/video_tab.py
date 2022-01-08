from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QEventLoop

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from facenet_pytorch import MTCNN

from model.triplet.model import TripletNet
from backend.tab_widget.video_thread import VideoWorkerThread


class VideoTab(QtWidgets.QWidget):
    def __init__(self, parent, bucket, database, user_id):
        super(VideoTab, self).__init__(parent)

        # Set up database
        self.bucket = bucket
        self.database = database
        self.user_id = user_id

        # Model set up
        self.model = TripletNet(pretrained=False, out_dim=256).eval()
        self.model.load_state_dict(torch.load("weight/model.pt", map_location="cpu"))
        self.face_detect = MTCNN(select_largest=False, post_process=False, device="cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((140, 140)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.video_display_label = QtWidgets.QLabel()
        self.video_display_label.setFixedSize(500, 500)

        self.start_button = QtWidgets.QPushButton("Start Video")
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

        self.stop_button = QtWidgets.QPushButton("Stop Video")
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

        self.take_frame_button = QtWidgets.QPushButton("Capture")
        self.take_frame_button.clicked.connect(self.handle_take_frame)
        self.take_frame_button.setEnabled(False)
        self.take_frame_button.setStyleSheet(
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
        side_panel_v_box.addWidget(self.start_button)
        side_panel_v_box.addWidget(self.stop_button)
        side_panel_v_box.addWidget(self.take_frame_button)

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
        self.take_frame_button.setEnabled(True)
        self.take_frame_button.repaint()

        video_file = 0
        self.video_thread_worker = VideoWorkerThread(self, video_file)

        self.video_thread_worker.frame_data_updated.connect(self.update_video_frames)
        self.video_thread_worker.start()

    def stop_current_video(self):
        if self.thread_is_running:
            self.thread_is_running = False
            self.video_thread_worker.stop_thread()

            self.video_display_label.clear()
            self.start_button.setEnabled(True)
            self.take_frame_button.setEnabled(False)

    def update_video_frames(self, video_frame):
        height, width, channels = video_frame.shape
        bytes_per_line = width * channels

        converted_Qt_image = QImage(video_frame.copy(), width, height, bytes_per_line, QImage.Format_RGB888)

        self.video_display_label.setPixmap(
            QPixmap.fromImage(converted_Qt_image).scaled(
                self.video_display_label.width(), self.video_display_label.height())
        )

    @staticmethod
    def q_sleep(time):
        loop = QEventLoop()
        QTimer.singleShot(time, loop.quit)
        loop.exec_()

    @staticmethod
    def pix_to_array(pixmap):
        h = pixmap.size().height()
        w = pixmap.size().width()

        q_image = pixmap.toImage()
        byte_str = q_image.bits().asstring(w * h * 4)

        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((h, w, 4))
        return img

    def handle_take_frame(self):
        self.take_frame_button.setEnabled(False)

        faces_array = []
        direction = ["ori", "top", "down", "right", "left"]

        for idx in range(5):
            self.parent().parent().parent().status_record.setText(f"Ready: {idx + 1}, {direction[idx]}")

            # Wait for two seconds for change poses
            self.q_sleep(2000)

            face = self.pix_to_array(self.video_display_label.pixmap())[:, :, :3]

            # Detect faces
            box, _ = self.face_detect.detect(Image.fromarray(face))
            if box is not None:
                box = box[0].astype("int").tolist()
                face = face[box[1]:box[3], box[0]:box[2]]

            faces_array.append(self.transform(face.copy()))

            self.parent().parent().parent().status_record.setText(f"Finish: {idx + 1}")

            # Wait for 1 seconds for another image
            self.q_sleep(1000)

        self.parent().parent().parent().status_record.setText("")

        # Transform into tensor and get features
        with torch.no_grad():
            features = self.model.get_features(torch.stack(faces_array)).mean(dim=0).tolist()

        # Upload features to firebase
        face_count = self.database.get(f"users/{self.user_id}", "face_count") + 1
        self.database.put(f"users/{self.user_id}", "face_count", face_count)
        self.database.put(f"users/{self.user_id}/identity", f"face{face_count}", features)

        self.take_frame_button.setEnabled(True)
