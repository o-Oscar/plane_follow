import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial
from PIL import Image
from PyQt5.QtCore import QEvent, QObject, QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QGridLayout,
)

from camera_commons import CameraMovementModel

"""
python -m camera_range.explorer

Permet de montrer l'enveloppe de vol dans laquelle le modèle est bien détecté en fonction de sa taille et du niveau de zoom de la caméra.
Donne une idée de ce qu'il est possible / impossible de faire
"""


class EventEater(QObject):
    def __init__(self):
        super().__init__()
        self.keypressed = None

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.keypressed = event.key()
        return False


class CustomSlider(QWidget):
    def __init__(self, slider_min, slider_max, text_format, callback):
        super().__init__()
        self.text_format = text_format
        self.callback = callback

        layout = QHBoxLayout()

        self.label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)

        self.label.setMinimumWidth(200)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self.value_changed)
        self.update_text()

    def value_changed(self, value):
        self.update_text()
        self.callback()

    def update_text(self):
        self.label.setText(self.text_format.format(self.slider.value()))


class MainLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)
        self.installEventFilter(self)

        self.px = None
        self.py = None
        self.tx = None
        self.ty = None
        self.d_max = 200
        self.d_min = 100

        self.get_image()

        self.size_to_px = 53 / 20  # 53 pixel for 20 meter

        self.setMinimumSize(self.im.size[0], self.im.size[1])

    def get_image(self):
        self.im = Image.open(
            "/home/oscar/workspace/plane_follow/python/camera_range/sources/image_croped.png"
        )
        self.im = self.im.convert("RGB")
        data = self.im.tobytes("raw", "RGB")
        self.qImg = QImage(
            data,
            self.im.size[0],
            self.im.size[1],
            self.im.size[0] * 3,
            QImage.Format_RGB888,
        )

    def paintEvent(self, *args, **kwargs):
        painter = QPainter(self)

        painter.drawImage(QRect(0, 0, self.im.size[0], self.im.size[1]), self.qImg)

        if (
            self.px is not None
            and self.py is not None
            and self.tx is not None
            and self.ty is not None
        ):
            width = self.d_max - self.d_min
            s = (self.d_min + self.d_max) // 2
            ta = np.arctan2(self.py - self.ty, self.px - self.tx)
            alen = 270

            brush = QBrush(Qt.GlobalColor.red)
            pen = QPen(brush, width // 2)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.setOpacity(0.3)

            painter.drawArc(
                int(self.px - s / 2),
                int(self.py - s / 2),
                int(s),
                int(s),
                -int((90 + ta * 180 / np.pi - alen / 2) * 16),
                alen * 16,
            )

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.set_cur_position(event)  # .pos().x(), event.pos().y())
            if event.button() == Qt.MouseButton.RightButton:
                self.set_cur_target(event)  # .pos().x(), event.pos().y())
        if event.type() == QEvent.Type.MouseMove:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.set_cur_position(event)  # .pos().x(), event.pos().y())
            if event.buttons() & Qt.MouseButton.RightButton:
                self.set_cur_target(event)  # .pos().x(), event.pos().y())
        return False

    def set_cur_position(self, event):
        self.px = event.pos().x()
        self.py = event.pos().y()
        self.repaint()

    def set_cur_target(self, event):
        self.tx = event.pos().x()
        self.ty = event.pos().y()
        self.repaint()


def get_spacer():
    return QSpacerItem(
        10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Size checker")
        self.setGeometry(100, 100, 700, 700)

        self.zoom_model = CameraMovementModel()
        # print(self.zoom_model.zoom_factor(zoom))

        layout = QVBoxLayout()

        self.image_label = MainLabel()

        main_holder = QWidget()
        main_holder_layout = QGridLayout()
        main_holder.setLayout(main_holder_layout)
        main_holder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_holder_layout.addItem(get_spacer(), 0, 0)
        main_holder_layout.addWidget(self.image_label, 1, 1)
        main_holder_layout.addItem(get_spacer(), 2, 2)

        self.plane_size_slider = CustomSlider(
            50, 200, "plane size : {} cm", self.update_image_label
        )
        self.zoom_slider = CustomSlider(0, 16244, "zoom : {}", self.update_image_label)

        layout.addWidget(main_holder, 100)
        layout.addWidget(self.plane_size_slider, 0)
        layout.addWidget(self.zoom_slider, 0)

        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

        self.installEventFilter(self)

        self.update_image_label()

        self.show()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key.Key_Q:
                self.close()
        return False

    def update_image_label(self):
        min_l = 40  # pix
        max_l = 200  # pix
        s = self.plane_size_slider.slider.value() / 100
        zoom = self.zoom_slider.slider.value()
        self.image_label.d_min = self.get_dist(max_l, s, zoom)
        self.image_label.d_max = self.get_dist(min_l, s, zoom)
        self.image_label.repaint()

    def get_dist(self, l, s, zoom):
        n = l * self.zoom_model.zoom_factor(zoom)
        alpha = n * np.pi / 2328
        d = s / alpha
        return d


if __name__ == "__main__":
    app = QApplication.instance()
    if not app:  # sinon on crée une instance de QApplication
        app = QApplication(sys.argv)

    window = MainWindow()

    app.exec()
