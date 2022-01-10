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
                self.capture = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.capture.connect((self.video_file.split(":")[-1], 9999))
                self.capture.settimeout(5)
            else:
                self.capture = cv2.VideoCapture(self.video_file)
        except Exception as e:
            self.parent.thread_is_running = False
            self.parent.start_button.setEnabled(True)
            self.parent.video_display_label.setText(str(e))
            self.capture.close()

        fps, start_time, record_fps_frames, count_frame = 0, 0, 20, 0
        while self.parent.thread_is_running:
            # Read frames from the camera
            try:
                if self.check_is_udp():
                    length = self.recv_all(self.capture, 16)
                    img_string = self.recv_all(self.capture, int(length))
                    frame = np.fromstring(img_string, dtype="uint8")
                    frame = cv2.imdecode(frame, 1)
                else:
                    ret_val, frame = self.capture.read()
            except (socket.timeout, BlockingIOError) as e:
                self.parent.thread_is_running = False
                self.parent.start_button.setEnabled(True)
                self.parent.video_display_label.setText(str(e))
                break

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
                self.capture.send(b"get")

        if self.check_is_udp():
            self.capture.close()
        else:
            self.capture.release()

    @staticmethod
    def recv_all(connect, count):
        buffer = b""
        while count:
            new_buffer = connect.recv(count)
            if not new_buffer:
                return None
            buffer += new_buffer
            count -= len(new_buffer)
        return buffer

    def check_is_udp(self):
        return "udp" in str(self.video_file)

    def stop_thread(self):
        self.wait()
        QtWidgets.QApplication.processEvents()
