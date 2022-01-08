import sys
from PyQt5 import QtWidgets

from firebase import firebase
from google.cloud import storage

from backend.login import Login
from backend.tab_widget import VideoTab, InfoTab, IdentityTab, MatplotlibTab, MonitorTab


class VLine(QtWidgets.QFrame):
    """VLine taken from
    https://stackoverflow.com/a/57944421/12751554
    """

    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, bucket, database, parent=None):
        super(TabWidget, self).__init__(parent)
        self.bucket = bucket
        self.database = database

        self.id = None

    def init_ui(self, user_id):
        self.video_container = VideoTab(self, self.bucket, self.database, user_id)
        self.info_container = InfoTab(self, self.bucket, self.database, user_id)
        self.identity_container = IdentityTab(self, self.bucket, self.database, user_id)
        self.spatial_info = MatplotlibTab(self, self.bucket, self.database, user_id)
        self.monitor_info = MonitorTab(self)

        self.addTab(self.video_container, "Face update")
        self.addTab(self.info_container, "Information")
        self.addTab(self.identity_container, "Recognized faces")
        self.addTab(self.spatial_info, "Spatial information")
        self.addTab(self.monitor_info, "Monitor")

    def closeEvent(self, event):
        if self.video_container.thread_is_running:
            self.video_container.video_thread_worker.quit()

    def log_out(self):
        if self.video_container.thread_is_running:
            self.video_container.video_thread_worker.quit()
        self.close()
        if self.parent().login_window.exec_() == QtWidgets.QDialog.Accepted:
            self.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bucket, database):
        super(MainWindow, self).__init__()

        self.login_window = Login(database)
        self.main_widget = TabWidget(bucket, database, parent=self)
        self.user_id = None

        self.status_id = QtWidgets.QLabel()
        self.status_id.setStyleSheet("border: none")
        self.status_record = QtWidgets.QLabel()
        self.status_record.setStyleSheet("border: none")
        self.log_out_button = QtWidgets.QPushButton("Log out")
        self.log_out_button.clicked.connect(self.main_widget.log_out)
        self.log_out_button.setStyleSheet(
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

        self.statusBar().showMessage("Status bar")
        self.statusBar().setStyleSheet('border: 0; background-color: #FFF8DC;')
        self.statusBar().setStyleSheet("QStatusBar::item {border: none;}")
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.status_record)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.status_id)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.log_out_button)

        if self.login_window.exec_() == QtWidgets.QDialog.Accepted:
            self.user_id = self.login_window.user_id
            self.main_widget.init_ui(self.user_id)
            self.init_ui()
            self.status_id.setText(f"ID: {self.user_id}")
        else:
            self.login_window.close()
            self.close()

    def init_ui(self):
        self.setCentralWidget(self.main_widget)
        self.show()


if __name__ == '__main__':
    import os

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "<GCP JSON file>"

    # Add google cloud client
    client = storage.Client()
    bucket = client.get_bucket("face_identity")

    database = firebase.FirebaseApplication(
        "<your url>", None
    )

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(bucket=bucket, database=database)
    app.exec_()
