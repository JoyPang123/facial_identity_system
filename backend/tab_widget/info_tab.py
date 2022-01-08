import io

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap

from backend.tab_widget.item import CustomItem


class InfoTab(QtWidgets.QWidget):
    def __init__(self, parent, bucket, database, user_id):
        super(InfoTab, self).__init__(parent)

        # Set up database
        self.bucket = bucket
        self.database = database
        self.user_id = user_id
        self.optimizer = torch.optim.Adam(self.parent().video_container.model.parameters(), lr=3e-4)

        self.monitor = QtWidgets.QLabel()
        self.monitor.setFixedSize(500, 500)

        self.reload_btn = QtWidgets.QPushButton()
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
        self.reload_btn.setText("reload list")
        self.reload_btn.clicked.connect(self.reload_list)
        self.delete_info_btn = QtWidgets.QPushButton()
        self.delete_info_btn.setStyleSheet(
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
        self.delete_info_btn.setText("delete")
        self.delete_info_btn.clicked.connect(self.delete_selected_info)
        self.list_view = QtWidgets.QListWidget()
        self.list_view.itemClicked.connect(self.list_item_selected)
        self.list_view.setStyleSheet(
            """
                QListWidget::item {
                    border: 0.5px solid gray;
                }
                QListWidget::item:selected {
                    border: 2px solid black;
                }
            """
        )
        secure_info = self.database.get(f"/users/{self.user_id}", "secure")

        if secure_info is not None:
            for keys, values in secure_info.items():
                custom_widget = CustomItem(self)
                custom_widget_item = QtWidgets.QListWidgetItem(self.list_view)
                custom_widget_item.setSizeHint(QSize(100, 100))
                custom_widget.set_pass_info(values["pass"], values["name"])
                custom_widget.set_time_info(keys)

                self.list_view.addItem(custom_widget_item)
                self.list_view.setItemWidget(custom_widget_item, custom_widget)

        side_h_box = QtWidgets.QHBoxLayout()
        side_h_box.addWidget(self.reload_btn)
        side_h_box.addWidget(self.delete_info_btn)
        side_v_box = QtWidgets.QVBoxLayout()
        side_v_box.addLayout(side_h_box)
        side_v_box.addWidget(self.list_view)
        main_h_box = QtWidgets.QHBoxLayout()
        main_h_box.addWidget(self.monitor, 1)
        main_h_box.addLayout(side_v_box)

        self.setLayout(main_h_box)

    def reload_list(self):
        self.list_view.clear()

        secure_info = self.database.get(f"/users/{self.user_id}", "secure")

        if secure_info is not None:
            for keys, values in secure_info.items():
                custom_widget = CustomItem()
                custom_widget_item = QtWidgets.QListWidgetItem(self.list_view)
                custom_widget_item.setSizeHint(QSize(100, 100))
                custom_widget.set_pass_info(values["pass"], values["name"])
                custom_widget.set_time_info(keys)

                self.list_view.addItem(custom_widget_item)
                self.list_view.setItemWidget(custom_widget_item, custom_widget)

        self.monitor.clear()

    def delete_selected_info(self):
        if len(self.list_view.selectedItems()) != 0:
            file_name = self.list_view.itemWidget(self.list_view.selectedItems()[0]).time_info.text()
            self.database.delete(f"/users/{self.user_id}/secure", file_name)
            blob = self.bucket.blob(file_name)
            try:
                blob.delete()
            except Exception:
                pass
            self.list_view.takeItem(self.list_view.selectionModel().selectedRows()[0].row())
            self.monitor.clear()

    def list_item_selected(self, item):
        try:
            file_name = item.listWidget().itemWidget(item).time_info.text()
            blob = self.bucket.blob(file_name)

            # Create temp_file_bytes
            temp_file_bytes = io.BytesIO()
            blob.download_to_file(temp_file_bytes)

            image = np.frombuffer(temp_file_bytes.getbuffer(), dtype="uint8")
            image = cv2.imdecode(image, flags=1)
            height, width, channels = image.shape
            bytes_per_line = width * channels
            converted_Qt_image = QImage(image, width, height, bytes_per_line, QImage.Format_RGB888)

            self.monitor.setPixmap(
                QPixmap.fromImage(converted_Qt_image).scaled(
                    self.monitor.width(), self.monitor.height())
            )
        except Exception as e:
            self.monitor.setText("No image")
