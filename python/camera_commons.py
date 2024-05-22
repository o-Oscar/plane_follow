import cv2
import numpy as np
import serial
import time
import traceback
import logging
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMainWindow,
    QLabel,
    QSlider,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, Qt, QEvent
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

"""
Interface de base pour piloter la caméra.
Surcharger la classe CameraDriver pour utiliser toutes les fonctionnalités. 
Ce fichier dispose de deux exemples en bas du script. 
"""


class CameraMovementModel:
    def __init__(self):
        # speed_model
        self.cmd_to_speed = np.array(
            [
                (-17, -862.6984362930882),
                (-16, -798.505174457466),
                (-15, -734.3119126218437),
                (-14, -670.1186507862215),
                (-13, -605.9253889505992),
                (-12, -541.732127114977),
                (-11, -477.53886527935464),
                (-10, -413.3456034437323),
                (-9, -349.15234160811),
                (-8, -284.95907977248777),
                (-7, -220.76581793686552),
                (-6, -156.5725561012432),
                (-5, -92.3792942656209),
                (-4, -54.293199705017045),
                (-3, -38.5378572448858),
                (-2, -30.415686807594664),
                (-1, -22.29351637030353),
                (0, 0),
                (1, 22.29351637030353),
                (2, 30.415686807594664),
                (3, 38.5378572448858),
                (4, 54.293199705017045),
                (5, 92.3792942656209),
                (6, 156.5725561012432),
                (7, 220.76581793686552),
                (8, 284.95907977248777),
                (9, 349.15234160811),
                (10, 413.3456034437323),
                (11, 477.53886527935464),
                (12, 541.732127114977),
                (13, 605.9253889505992),
                (14, 670.1186507862215),
                (15, 734.3119126218437),
                (16, 798.505174457466),
                (17, 862.6984362930882),
            ]
        )
        self.speed_midpoints = (
            self.cmd_to_speed[1:, 1] + self.cmd_to_speed[:-1, 1]
        ) / 2
        self.speed_midpoints = np.concatenate([[-np.inf], self.speed_midpoints])

        # old_zoom_model
        self.wide_zoom = 990
        self.wide_move = 652
        self.narrow_zoom = 12137
        self.narrow_move = 125

        self.zoom_fac_base = self.wide_move / self.narrow_move
        self.zoom_fac_exponent = self.wide_zoom - self.narrow_zoom
        self.zoom_fac_a = (
            self.wide_move
            / 640
            / np.power(self.zoom_fac_base, self.wide_zoom / self.zoom_fac_exponent)
        )

        # new_zoom_model
        self.zoom_coefs = np.array(
            [-5.976987544816842e-09, -7.022266302305842e-05, 0.1064226581903614]
        )

    def bit_speed_to_cmd(self, target_speed):
        first_id = np.max(np.where(self.speed_midpoints < target_speed))
        return int(self.cmd_to_speed[first_id, 0])

    def zoom_factor_old(self, zoom):
        return self.zoom_fac_a * np.power(
            self.zoom_fac_base, zoom / self.zoom_fac_exponent
        )

    def zoom_factor(self, zoom):
        return np.exp(
            self.zoom_coefs[0] * zoom**2
            + self.zoom_coefs[1] * zoom
            + self.zoom_coefs[2]
        )

    def px_speed_to_cmd(self, target_speed_px, zoom):
        return self.bit_speed_to_cmd(target_speed_px * self.zoom_factor(zoom))

    def cmd_to_bit_speed(self, cmd):
        return self.cmd_to_speed[int(cmd - self.cmd_to_speed[0, 0]), 1]

    def cmd_to_px_speed(self, cmd, zoom):
        return self.cmd_to_bit_speed(cmd) / self.zoom_factor(zoom)

    def update(self):
        # advance the cmds in the virtual serial port buffer
        # simulate the actual speed using the calibrated maximum acceleration
        raise NotImplementedError("please implement !")

    def send_speed_cmd(self, cmd):
        # set the last speed cmd that has been sent
        raise NotImplementedError("please implement !")


class ImgViewerLabel(QLabel):
    def __init__(self, debug_label, click_callback):
        super().__init__()
        self.debug_label = debug_label
        self.setMouseTracking(True)
        self.img = None
        self.click_callback = click_callback

    def imshow(self, img):
        self.img = img
        qImg = QImage(img.data, 640, 480, 3 * 640, QImage.Format_RGB888)
        self.setPixmap(QPixmap(qImg))

    def get_img_debug_text(self, QMouseEvent):
        pos = QMouseEvent.pos()
        if (
            self.img is not None
            and pos.x() < self.img.shape[1]
            and pos.y() < self.img.shape[0]
        ):
            r, g, b = self.img[pos.y(), pos.x()]
        else:
            r, g, b = (np.nan, np.nan, np.nan)

        return "x: {}, y: {}, rgb: ({}, {}, {})".format(pos.x(), pos.y(), r, g, b)

    def mouseMoveEvent(self, QMouseEvent):
        self.debug_label.setText(self.get_img_debug_text(QMouseEvent))

    def mousePressEvent(self, QMouseEvent):
        print("mouse pressed ", self.get_img_debug_text(QMouseEvent))
        self.click_callback(QMouseEvent.pos().x(), QMouseEvent.pos().y())


class MyWindow(QMainWindow):
    def __init__(self, name, click_callback):
        super().__init__()
        self.setWindowTitle(name)
        self.setGeometry(100, 100, 700, 700)

        layout = QVBoxLayout()

        self.debug_label = QLabel()
        self.img_viewer = ImgViewerLabel(self.debug_label, click_callback)
        layout.addWidget(self.debug_label)
        layout.addWidget(self.img_viewer)

        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

    def on_button_click(self):
        print(f"Button clicked!")


class EventEater(QObject):
    def __init__(self):
        super().__init__()
        self.keypressed = None

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.keypressed = event.key()
        return False


class ValueSlider(QWidget):
    def __init__(self, window, name, value, track_min, track_max, trackbar_data):
        super().__init__()

        self.trackbar_data = trackbar_data
        self.name = name
        self.txt_fmt = "{} : {}"

        window.central_widget.layout().addWidget(self)

        layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setText(self.txt_fmt.format(name, value))
        self.label.setFixedWidth(200)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(track_min)
        self.slider.setMaximum(track_max)
        self.slider.setValue(value)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self.valueChanged)

    def valueChanged(self):
        self.label.setText(self.txt_fmt.format(self.name, self.slider.value()))
        self.trackbar_data[self.name] = self.slider.value()


class QtHandle:
    def __init__(self, click_callback):
        self.app = QApplication.instance()
        if not self.app:  # sinon on crée une instance de QApplication
            self.app = QApplication(sys.argv)

        self.windows = {}
        self.event_eater = EventEater()
        self.click_callback = click_callback

    def namedWindow(self, name):
        fen = MyWindow(name, self.click_callback)
        self.windows[name] = fen
        fen.installEventFilter(self.event_eater)
        fen.show()

    def imshow(self, name, cv2_img):
        if name not in self.windows:
            self.namedWindow(name)

        self.windows[name].img_viewer.imshow(cv2_img)

    def waitKey(self, delay):
        self.event_eater.keypressed = None
        self.app.processEvents()
        return self.event_eater.keypressed

    def destroyAllWindows(self):
        print("exiting qt")
        self.app.quit()
        for win in self.windows.values():
            win.close()

    def create_trackbar(
        self, window_name, name, value, track_min, track_max, trackbar_data
    ):
        ValueSlider(
            self.windows[window_name],
            name,
            value,
            track_min,
            track_max,
            trackbar_data,
        )


class FPSCounter:
    def __init__(self):
        self.keep_delay = 1  # in secound
        self.data = []

    def wake_up(self):
        ct = time.time()
        self.data.append(ct)
        self.data = [x for x in self.data if x > ct - self.keep_delay]

    @property
    def fps(self):
        return len(self.data) / self.keep_delay

    def __str__(self):
        return "fps : {}".format(int(self.fps))


def cv_print(img, text):
    img = img[:]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    # fontColor = (0, 0, 0)
    fontColor = np.array([254, 254, 254]).astype(int)

    lineType = 1
    cv2.putText(
        img,
        text,
        (0, img.shape[0] - 10),
        font,
        fontScale,
        [int(fontColor[i]) for i in range(3)],
        3,
        lineType,
        bottomLeftOrigin=False,
    )

    mask = cv2.inRange(img, fontColor, fontColor)
    dilated = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    res_mask = np.logical_or(mask, np.logical_not(dilated))
    img = cv2.bitwise_and(img, img, mask=(res_mask * 255).astype(np.uint8))

    return img


class CameraDriver:
    def __init__(self):

        self.qt_handle = QtHandle(self.click_callback)

        self.cap = cv2.VideoCapture(2)
        if self.cap.isOpened() == False:
            raise ValueError("Error opening video stream or file")

        self.arduino = serial.Serial(
            port="/dev/ttyACM0", baudrate=38400, timeout=0  # 1 / 26 / 2
        )
        time.sleep(1)  # delay for the port to open properly
        self.arduino.write(bytearray([0x81, 0x01, 0x04, 0x06, 0x03, 0xFF]))

        # fmt: off
        # camera factors
        self.wide_angle = 72 # in degrees
        self.image_width = 640 # in pixel
        self.image_height = 480 # in pixel
        self.zoom_fac = 1.0009853506942101 # in bit^-1
        self.wide_speed_fac = 12 # in pix / sec / bit
        self.wide_pos_fac = 0.9 # in pix / bit

        self.pan_m = -2700 # theoretically it is 0xE1E5 - 0x10000
        self.pan_M = 2700 # theoretically it is 0x1E1B
        self.tilt_m = -230 # theoretically it is 0xFC75 - 0x10000
        self.tilt_M = 1200 # 0x0FF0
        self.zoom_m = 0
        self.zoom_M = 16244 # theoretically it is 0xFFFF

        # fmt: on

        # utils
        self.fps_counter = FPSCounter()
        self.window_name = "main_window"
        self.trackbar_data = {}
        self.timer_data = {}
        self.read_buffer = bytearray([])

        self.cur_pan = 0
        self.cur_tilt = 0
        self.cur_zoom = 0
        self.pos_update_time = None
        self.zoom_update_time = None
        self.init_pos = [0, 0]

        self.frame_id = 0
        self.using_plt = False

        self.root_save_path = Path("results/frames")
        self.save_frames = False
        self.frames_save_path = None

        self.movement_model = CameraMovementModel()

        # cv2.namedWindow(self.window_name)
        self.qt_handle.namedWindow(self.window_name)

    def main_loop(self):
        if self.using_plt:
            plt.ion()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self.fps_counter.wake_up()
            if ret == True:

                keypressed = self.qt_handle.waitKey(1)
                if self.using_plt:
                    plt.pause(0.000001)

                self.update_read_buffer()
                try:
                    frame = np.stack(  # frame from bgr to rgb
                        [frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]], axis=2
                    )
                    to_show, keep_going = self.frame_callback(frame, keypressed)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    break

                if not keep_going:
                    break

                self.save_cur_frame(frame)

                to_show = cv_print(to_show, str(self.fps_counter.fps))
                self.qt_handle.imshow(self.window_name, to_show)

                # Press Q on keyboard to  exit
                if keypressed == Qt.Key_Q:
                    break

                self.frame_id += 1

            else:
                break

        if self.using_plt:
            plt.ioff()

        # When everything done, release the video capture object
        self.cap.release()

        # Closes all the frames
        # cv2.destroyAllWindows()
        self.qt_handle.destroyAllWindows()

        self.send_speed_cmd([0, 0])

    def create_save_path(self):
        if self.frames_save_path is not None:
            return
        for i in range(10000):
            self.frames_save_path = self.root_save_path / "exp_{:04d}".format(i)
            if not self.frames_save_path.exists():
                self.frames_save_path.mkdir(parents=True)
                break
        print("Saving frames at ", self.frames_save_path)

    def save_cur_frame(self, frame):
        if not self.save_frames:
            return
        if self.frames_save_path is None:
            self.create_save_path()
        pil_img = Image.fromarray(frame)
        pil_img.save(self.frames_save_path / "{:06d}.jpg".format(self.frame_id))

    def frame_callback(self, frame, keypressed):
        raise NotImplementedError("please implement the frame callback")

    def click_callback(self, x, y):
        print("click callback not implemented")

    def pos_to_bytes(self, target, t_min, t_max):
        target = min(max(target, t_min), t_max)
        zb3 = target % 0x10
        zb2 = (target // 0x10) % 0x10
        zb1 = (target // 0x100) % 0x10
        zb0 = (target // 0x1000) % 0x10
        return [zb0, zb1, zb2, zb3]

    def bytes_to_pos(self, bytes):
        to_return = bytes[0] * 0x1000 + bytes[1] * 0x100 + bytes[2] * 0x10 + bytes[3]
        if to_return > 0x8000:
            to_return = to_return - 0x10000
        return to_return

    def send_opt_cmd(self, target_speed, target_zoom):
        sign = np.sign(target_speed).astype(int)
        abs_speed = np.abs(target_speed).astype(np.uint8)
        speed_bytes = np.clip(abs_speed, 1, 0x17)
        pan_dir = 0x03 if abs_speed[0] == 0 else (0x02 if sign[0] > 0 else 0x01)
        tilt_dir = 0x03 if abs_speed[1] == 0 else (0x02 if sign[1] > 0 else 0x01)

        self.arduino.write(
            bytearray(
                [0xAA]
                + [speed_bytes[0], speed_bytes[1], pan_dir, tilt_dir]
                + self.pos_to_bytes(target_zoom, self.zoom_m, self.zoom_M)
                + [0xFF]
            )
        )

    def send_zoom_cmd(self, target_zoom):
        self.arduino.write(
            bytearray(
                [0x81, 0x01, 0x04, 0x47]
                + self.pos_to_bytes(target_zoom, self.zoom_m, self.zoom_M)
                + [0xFF]
            )
        )
        # res = self.arduino.read_all()
        # print("0x" + " 0x".join(format(x, "02x") for x in res))

    def send_speed_cmd(self, target_speed):
        sign = np.sign(target_speed).astype(int)
        abs_speed = np.abs(target_speed).astype(np.uint8)
        speed_bytes = np.clip(abs_speed, 1, 0x17)
        pan_dir = 0x03 if abs_speed[0] == 0 else (0x02 if sign[0] > 0 else 0x01)
        tilt_dir = 0x03 if abs_speed[1] == 0 else (0x02 if sign[1] > 0 else 0x01)

        self.arduino.write(
            bytearray(
                [0x81, 0x01, 0x06, 0x01]
                + [speed_bytes[0], speed_bytes[1], pan_dir, tilt_dir]
                + [0xFF]
            )
        )

    def send_pos_cmd(self, target_pos, target_speed):
        abs_speed = np.abs(target_speed).astype(np.uint8)
        speed_bytes = np.clip(abs_speed, 1, 0x17)

        self.arduino.write(
            bytearray(
                [0x81, 0x01, 0x06, 0x02]
                + [speed_bytes[0], speed_bytes[1]]
                + self.pos_to_bytes(target_pos[0], self.pan_m, self.pan_M)
                + self.pos_to_bytes(target_pos[1], self.tilt_m, self.tilt_M)
                + [0xFF]
            )
        )

    def wait_for_response(self, resp_start):
        bytes = self.arduino.read(len(resp_start))
        found_start = False
        for i in range(20):
            if bytes[: len(resp_start)] == resp_start:
                found_start = True
                break
            else:
                bytes = bytes[1:]
                if len(bytes) < len(resp_start):
                    bytes = bytes + self.arduino.read(1)
        if not found_start:
            return [], False

        found_end = False
        for i in range(20):
            if bytes[-1] == 0xFF:
                found_end = True
                break
            else:
                bytes = bytes + self.arduino.read(1)

        return bytes, found_end and found_start

    def send_pos_inq_cmd(self):
        self.arduino.write(bytearray([0x81, 0x09, 0x06, 0x12, 0xFF]))

    def send_zoom_inq_cmd(self):
        self.arduino.write(bytearray([0x81, 0x09, 0x04, 0x47, 0xFF]))

    def update_read_buffer(self):
        self.read_buffer = self.read_buffer + self.arduino.readall()
        # self.inquiry_dict = {}

        last_start_byte = 0
        for i in range(len(self.read_buffer)):
            if self.read_buffer[i] == 0xFF:
                self.handle_msg(self.read_buffer[last_start_byte : i + 1])
                last_start_byte = i + 1

        self.read_buffer = self.read_buffer[last_start_byte:]

    def handle_msg(self, msg):
        # TODO : add zoom read handling
        # print("0x" + " 0x".join(format(x, "02x") for x in msg))
        # test for pos inquiry response
        if len(msg) >= 11 and msg[-11:-9] == bytearray([0x90, 0x50]):
            msg = msg[-11:]
            self.cur_pan = self.bytes_to_pos(msg[2:6])
            self.cur_tilt = self.bytes_to_pos(msg[6:10])
            self.pos_update_time = time.time()

        if len(msg) >= 7 and msg[-7:-5] == bytearray([0x90, 0x50]):
            msg = msg[-7:]
            self.cur_zoom = self.bytes_to_pos(msg[2:6])
            self.zoom_update_time = time.time()

    def stop_camera_movement(self):
        self.arduino.write(
            bytearray([0x81, 0x01, 0x06, 0x01, 0x01, 0x01, 0x03, 0x03, 0xFF])
        )

    # what the camera returns is usually nonsense due to the way software serial handles interrupts
    def clear_movement_buffer(self):
        # self.arduino.write(bytearray([0x88, 0x01, 0x00, 0x01, 0xFF]))
        # self.arduino.write(bytearray([0x81, 0x21, 0xFF]))
        # self.arduino.write(bytearray([0x81, 0x22, 0xFF]))

        return self.arduino.read_all()
        # print("0x" + " 0x".join(format(x, "02x") for x in res))

    def add_trackbar(self, name, value, track_min, track_max):
        self.trackbar_data[name] = value
        # cv2.createTrackbar(
        #     name,
        #     self.window_name,
        #     value - track_min,
        #     track_max - track_min,
        #     lambda x: self.trackbar_data.update({name: x + track_min}),
        # )
        self.qt_handle.create_trackbar(
            self.window_name, name, value, track_min, track_max, self.trackbar_data
        )

    def timer(self, delay):  # delay in second
        if delay not in self.timer_data:
            self.timer_data[delay] = 0

        ct = time.time()
        if self.timer_data[delay] + delay <= ct:
            self.timer_data[delay] = max(self.timer_data[delay] + delay, ct - delay / 2)
            return True
        else:
            return False


class CustomDriver(CameraDriver):
    def __init__(self):
        super().__init__()

        # example of slider callback
        self.add_trackbar("pan", 0, self.pan_m, self.pan_M)
        self.add_trackbar("tilt", 0, self.tilt_m, self.tilt_M)
        # self.add_trackbar("tilt", 0, self.tilt_m // 8, self.tilt_M // 8)
        self.add_trackbar("zoom", 0, self.zoom_m, self.zoom_M)

    # example of simple callbacks
    def frame_callback(self, frame, keypressed):

        # using the trackbars values
        target_pos = [
            self.trackbar_data["pan"],
            self.trackbar_data["tilt"],
        ]
        target_speed = [0x20, 0x20]  # max speed

        # setting camera target
        if self.frame_id % 2 == 0:
            self.send_pos_cmd(target_pos, target_speed)
        if self.frame_id % 2 == 1:
            self.send_zoom_cmd(self.trackbar_data["zoom"])

        self.send_pos_inq_cmd()
        self.send_zoom_inq_cmd()
        print(self.cur_pan, self.cur_tilt, self.cur_zoom)

        if keypressed == Qt.Key_Space:
            print("spacebar has been pressed")

        return frame, True


class CustomOptDriver(CameraDriver):
    def __init__(self):
        super().__init__()

        self.add_trackbar("pan", 0, -8, 8)
        self.add_trackbar("tilt", 0, -8, 8)
        # self.add_trackbar("tilt", 0, self.tilt_m // 8, self.tilt_M // 8)
        self.add_trackbar("zoom", 0, self.zoom_m, self.zoom_M)

        self.is_running = False

    # example of simple callbacks
    def frame_callback(self, frame, keypressed):

        # using the trackbars values
        target_speed = [
            self.trackbar_data["pan"],
            self.trackbar_data["tilt"],
        ]

        # setting camera target
        if self.is_running:
            if self.frame_id % 10 == 0:
                self.send_zoom_cmd(self.trackbar_data["zoom"])
            else:
                self.send_speed_cmd(target_speed)
                self.send_pos_inq_cmd()
                self.send_zoom_inq_cmd()
            # self.send_opt_cmd(target_speed, self.trackbar_data["zoom"])
        else:
            self.send_pos_cmd([0, 0], [8, 8])

        if keypressed == Qt.Key_Space:
            print("spacebar has been pressed")
            self.is_running = not self.is_running

        print(self.cur_pan, self.cur_tilt, self.cur_zoom)

        return frame, True


if __name__ == "__main__":
    if True:
        camera_driver = CustomDriver()
        camera_driver.main_loop()
    else:
        camera_driver = CustomOptDriver()
        camera_driver.main_loop()
