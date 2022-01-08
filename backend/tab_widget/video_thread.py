import base64
import time

import cv2
import numpy as np
import imutils
import socket
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal


class VideoWorkerThread(QThread):
    frame_data_updated = pyqtSignal(np.ndarray)

    def __init__(self, parent, video_file=None):
        super().__init__()
        self.parent = parent
        self.video_file = video_file

    def run(self):
        try:
            if self.check_is_udp():
                BUFF_SIZE = 65536
                self.capture = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.capture.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
                self.capture.sendto(b"connect", (self.video_file.split(":")[-1], 9999))
                self.capture.settimeout(5)
            else:
                self.capture = cv2.VideoCapture(self.video_file)
        except Exception as e:
            self.parent.thread_is_running = False
            self.parent.start_button.setEnabled(True)
            self.parent.video_display_label.setText(str(e))
            return

        fps, start_time, record_fps_frames, count_frame = 0, 0, 20, 0
        while self.parent.thread_is_running:
            # Read frames from the camera
            try:
                if self.check_is_udp():
                    packet = self.capture.recv(BUFF_SIZE)
                    data = base64.b64decode(packet, ' /')
                    frame = np.fromstring(data, dtype=np.uint8)
                    frame = cv2.imdecode(frame, 1)
                else:
                    ret_val, frame = self.capture.read()
            except socket.timeout as e:
                self.parent.thread_is_running = False
                self.parent.start_button.setEnabled(True)
                self.parent.video_display_label.setText(str(e))
                return

            frame = imutils.resize(frame, width=640)
            x, y = 80, 0
            frame = frame[y:y + 480, x:x + 480, :]

            frame = cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if count_frame == record_fps_frames:
                fps = round(record_fps_frames / (time.time() - start_time))
                start_time = time.time()
                count_frame = 0
            count_frame += 1

            self.frame_data_updated.emit(frame)

            if self.check_is_udp():
                self.capture.sendto(b"get", (self.video_file.split(":")[-1], 9999))

        if self.check_is_udp():
            self.capture.close()
        else:
            self.capture.release()

    def check_is_udp(self):
        return "udp" in str(self.video_file)

    def stop_thread(self):
        self.wait()
        QtWidgets.QApplication.processEvents()
