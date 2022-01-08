from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class IdentityTab(QtWidgets.QWidget):
    def __init__(self, parent, bucket, database, user_id):
        super(IdentityTab, self).__init__(parent)

        # Set up the database
        self.bucket = bucket
        self.database = database
        self.user_id = user_id

        self.identity_list = []

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)

        self.table.setHorizontalHeaderLabels(["Name", "Embedded"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(self.table.SelectRows)

        self.identity_list.clear()
        data = self.database.get(f"/users/{self.user_id}", "identity")
        if data is not None:
            self.table.setRowCount(len(data))
            for idx, (key, value) in enumerate(data.items()):
                name = QtWidgets.QTableWidgetItem(key)
                name.setTextAlignment(Qt.AlignCenter)
                embed = QtWidgets.QTableWidgetItem(str(value))

                self.table.setItem(idx, 0, name)
                self.table.setItem(idx, 1, embed)

                self.identity_list.append(key)

        self.table.itemChanged.connect(self.table_modify)

        self.reload_identity_btn = QtWidgets.QPushButton()
        self.reload_identity_btn.setText("reload")
        self.reload_identity_btn.clicked.connect(self.reload_identity)
        self.remove_identity_btn = QtWidgets.QPushButton()
        self.remove_identity_btn.setText("remove")
        self.remove_identity_btn.clicked.connect(self.remove_identity_row)

        side_v_box = QtWidgets.QVBoxLayout()
        side_v_box.addWidget(self.reload_identity_btn)
        side_v_box.addWidget(self.remove_identity_btn)
        main_h_box = QtWidgets.QHBoxLayout()
        main_h_box.addWidget(self.table, 1)
        main_h_box.addLayout(side_v_box)

        self.setLayout(main_h_box)

    def remove_identity_row(self):
        indices = self.table.selectionModel().selectedRows()
        indices = sorted(indices, reverse=True)

        remove_count = 0
        for index in indices:
            remove_count += 1
            name = self.table.item(index.row(), 0).text()
            self.database.delete(f"/users/{self.user_id}/identity", name)
            self.table.removeRow(index.row())
            self.identity_list.pop(index.row())

        if remove_count != 0:
            cur_face_count = self.database.get(f"/users/{self.user_id}", "face_count")
            self.database.put(f"/users/{self.user_id}", "face_count", cur_face_count - remove_count)

    def reload_identity(self):
        data = self.database.get(f"/users/{self.user_id}", "identity")
        self.table.itemChanged.disconnect()
        self.table.setRowCount(0)

        self.identity_list.clear()
        if data is not None:
            self.table.setRowCount(len(data))
            for idx, (key, value) in enumerate(data.items()):
                name = QtWidgets.QTableWidgetItem(key)
                name.setTextAlignment(Qt.AlignCenter)
                embed = QtWidgets.QTableWidgetItem(str(value))

                self.table.setItem(idx, 0, name)
                self.table.setItem(idx, 1, embed)
                self.identity_list.append(key)

        self.table.itemChanged.connect(self.table_modify)

    def table_modify(self, item):
        if item.column() == 0:
            old_name = self.identity_list[item.row()]
            name = self.table.item(item.row(), item.column()).text()
            value = self.database.get(f"/users/{self.user_id}/identity", old_name)
            self.database.delete(f"/users/{self.user_id}/identity", old_name)
            self.database.put(f"/users/{self.user_id}/identity", name, value)
            self.identity_list[item.row()] = name