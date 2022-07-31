from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np
import face_recognition
from PIL import Image

import sys
import os
import warnings
import sqlite3
import io

# change author.
name = "*******"


class FaceRecog:
    def __init__(self, cursor, con):
        self.known_encodings, self.known_names = [], []
        self.cursor, self.con = cursor, con
        self.process_this_frame = True

    def call_face(self):
        # database select query
        m = self.cursor.execute("""SELECT * FROM FACE""")
        self.con.commit()

        for x in m:
            name, ext = os.path.splitext(x[0])
            if ext == ".jpg":
                print(name)
                try:
                    self.known_names.append(name)
                    img = np.array(Image.open(io.BytesIO(x[1])))
                    face_encoding = face_recognition.face_encodings(img)[0]
                    self.known_encodings.append(face_encoding)
                except:
                    print("exception")

        self.locations = []
        self.face_encodings = []
        self.face_names = []

    def __del__(self):
        pass

    def get_frame(self, frame, mosaic):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if self.process_this_frame:
            self.locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(
                rgb_small_frame, self.locations
            )

            self.face_names = []
            self.face_dist = []
            for face_encoding in self.face_encodings:
                distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                min_value = min(distances)

                name = "NotFound"

                if min_value < 0.45:
                    idx = np.argmin(distances)
                    name = self.known_names[idx]

                self.face_names.append(name)
                self.face_dist.append(sum(distances) / len(distances))

        self.process_this_frame = not self.process_this_frame

        for (top, right, bottom, left), name, dist in zip(
            self.locations, self.face_names, self.face_dist
        ):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            font = cv2.FONT_HERSHEY_DUPLEX

            if name == "NotFound":
                # mosaic
                if mosaic:
                    face_img = frame[top:bottom, left:right]
                    face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.10, fy=0.10)
                    face_img = cv2.resize(
                        face_img,
                        (right - left, bottom - top),
                        interpolation=cv2.INTER_AREA,
                    )
                    frame[top:bottom, left:right] = face_img
                # generate box and text
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )
                cv2.putText(
                    frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
                )
            # advanced
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED
                )
                cv2.putText(
                    frame,
                    "%s(%d%s)" % (name, int(dist * 100), "%"),
                    (left + 6, bottom - 6),
                    font,
                    0.75,
                    (255, 255, 255),
                    1,
                )

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        _, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()


# Thread for video
class ShowVideo(QObject):
    # static value
    flag, mosaic_flag = 0, False
    VideoSignal = pyqtSignal(QImage)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # constructor
    def __init__(self, cursor, con):
        super(ShowVideo, self).__init__(parent=None)
        self.face_recog = FaceRecog(cursor, con)  # refer face_recog.py
        self.run_video = False

    @pyqtSlot()
    def startVideo(self):
        global image

        while self.run_video:
            _, image = self.camera.read()
            height, width = image.shape[:2]

            # flag
            if self.flag:
                image = self.face_recog.get_frame(image, self.mosaic_flag)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
            self.VideoSignal.emit(
                QImage(
                    color_swapped_image.data,
                    width,
                    height,
                    color_swapped_image.strides[0],
                    QImage.Format_RGB888,
                )
            )

            loop = QEventLoop()
            QTimer.singleShot(25, loop.quit)  # 25 ms
            loop.exec_()

    # click "BTN2" event, callback => face recognition
    @pyqtSlot()
    def face_detection(self):
        self.flag = ~self.flag

    # click "BTN3" event, callback  => mosaic flag set
    @pyqtSlot()
    def mosaic(self):
        self.mosaic_flag = ~self.mosaic_flag


# Thread2 for image view
class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

    def initUI(self):
        self.setWindowTitle("")

    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            pass

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


# widget setting for main window
class MainWidget(QWidget):
    def __init__(self):
        super().__init__(parent=None)

        # label
        manual_label = [
            QLabel("Face Recognition\nMosaic"),
            QLabel("Created by %s" % name),
        ]

        for i in range(0, 2):
            manual_label[i].setStyleSheet("Color : gray")
            manual_label[i].setFont(QFont("", 60 - (24 * i)))

        # horizontal layout
        hbox = [QHBoxLayout(), QHBoxLayout()]
        for i in range(0, 2):
            hbox[i].addStretch(1)
            hbox[i].addWidget(manual_label[i])
            hbox[i].addStretch(1)
        # btn
        self.btn_db = QPushButton("%s사진저장 DB%s" % (" " * 6, " " * 6))
        self.btn_db.setFont(QFont("", 30))
        self.btn_start = QPushButton("%sStart%s" % (" " * 12, " " * 12))
        self.btn_start.setFont(QFont("", 30))
        # vertical layout
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox[0])
        vbox.addLayout(hbox[1])
        vbox.addStretch(1)

        hbox1, hbox2 = QHBoxLayout(), QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(self.btn_db)
        hbox1.addStretch(1)

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_start)
        hbox2.addStretch(1)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        self.setLayout(vbox)


class MainWindow(QMainWindow):
    def __init__(self, cur, con):
        self.thread = QThread()
        self.vid = None
        self.image_viewer = None
        self.main_window2 = None

        super().__init__(parent=None)
        self.cursor = cur  # sqlite cursor
        self.con = con
        # if not exist face table, create table
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS FACE (name TEXT, data BLOB)"""
        )

        wg = MainWidget()
        self.setCentralWidget(wg)
        self.resize(800, 600)
        self.setWindowTitle("Project")
        self.show()
        wg.btn_db.clicked.connect(self.func_db)
        wg.btn_start.clicked.connect(self.func_camera)

    def func_db(self):
        fname = QFileDialog.getOpenFileName(
            self, "choose only image files"
        )  # file dialog
        try:
            name_list = fname[0].split("/")
            face_name = name_list[-1]
            with open(fname[0], "rb") as f:
                bin = f.read()  # binary
            # insert image
            self.cursor.execute(
                """INSERT INTO FACE (name, data) VALUES (?, ?)""", (face_name, bin)
            )
            self.con.commit()
        except:
            print("no file")

    def func_camera(self):
        self.vid = ShowVideo(cur, con)
        self.vid.face_recog.call_face()
        self.main_window2 = MainWindow2(self)  # second page instance
        self.main_window2.working()
        self.main_window2.btn_back.clicked.connect(self.back_main)
        self.main_window2.btn_face.clicked.connect(self.vid.face_detection)
        self.main_window2.btn_mos.clicked.connect(self.vid.mosaic)
        self.vid.VideoSignal.connect(self.image_viewer.setImage)
        self.vid.startVideo()

    # come back
    def back_main(self):
        self.vid.run_video = False
        self.show()
        self.main_window2.close()


class MainWindow2(QMainWindow):
    def __init__(self, previous_instance):
        super(MainWindow2, self).__init__()
        # GUI generation and option
        self.cursor = previous_instance
        self.btn_face = QPushButton("BTN2")
        self.btn_face.setFont(QFont("", 20))
        self.btn_mos = QPushButton("BTN3")
        self.btn_mos.setFont(QFont("", 20))
        self.btn_back = QPushButton("Back")
        self.btn_back.setFont(QFont("", 20))

        # widget
        vertical_layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()
        previous_instance.image_viewer = ImageViewer()
        horizontal_layout.addWidget(previous_instance.image_viewer)
        hbox = QHBoxLayout()
        vertical_layout.addLayout(horizontal_layout)  # 레이아웃
        hbox.addWidget(self.btn_face)
        hbox.addWidget(self.btn_mos)
        hbox.addWidget(self.btn_back)

        vertical_layout.addLayout(hbox)

        layout_widget = QWidget()
        layout_widget.setLayout(vertical_layout)

        # 2nd window
        self.setCentralWidget(layout_widget)
        self.setFixedSize(800, 600)
        self.setWindowTitle("Project")
        self.show()
        previous_instance.close()

        exit = QAction("Quit", self)
        exit.triggered.connect(self.closeEvent)

    def closeEvent(self, _):
        if self.cursor.vid.run_video:
            sys.exit()

    def working(self):
        self.cursor.thread.start()
        self.cursor.vid.run_video = True
        self.cursor.vid.moveToThread(self.cursor.thread)


if __name__ == "__main__":
    warnings.filterwarnings(action="ignore")

    # sqlite
    con = sqlite3.connect("./image.db")
    cur = con.cursor()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(cur, con)
    window.setFixedSize(800, 600)
    sys.exit(app.exec_())
