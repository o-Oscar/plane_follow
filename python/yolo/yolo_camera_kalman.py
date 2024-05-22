from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from ultralytics import YOLO

from camera_commons import CameraDriver
import time

"""
python -m yolo.yolo_camera_kalman

Le script principal qui fait tout tourner : 
    - Lecture des images de la caméra
    - Yolo pour détection l'avion
    - Les filtres de kalman pour estimer la position et la vitesse de l'avion
    - L'arduino pour faire bouger la caméra là où on a besoin

Utiliser la touche "t" quand l'avion est détecté pour demander à la caméra de tracker l'avion.
Utiliser la touche "Space" pour lancer / arrêter l'enregistrement de la vidéo (sous forme de frames à plat) pour être utilisée pour créer d'autres exemples d'entraînement et améliorer les performances de yolo.
"""


class KalmanFilter:
    def __init__(self, F, B, P0, Q, H, R):
        self.F = F
        self.B = B
        self.P0 = P0
        self.Q = Q
        self.H = H
        self.R = R
        self.n = self.F.shape[0]

        self.reset(np.zeros((self.n, 1)))

    def reset(self, x):
        self.P = self.P0
        self.x = x

    def update(self, u, z):  # z = 0 if we do not see the plane
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        if z is not None:
            y_tilde = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_tilde
            self.P = (np.eye(self.n) - K @ self.H) @ self.P


def get_cst_speed_kalman(params):
    dt = 1 / 25
    F = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [0]])
    P0 = np.diag([params["p"], params["p"]])
    Q = np.diag([params["q00"], params["q11"]])
    H = np.array([[1, 0]])
    R = np.diag([params["r"]])
    return KalmanFilter(F, B, P0, Q, H, R)


class PosAxisDriver:
    def __init__(self):
        kalman_params = {"q00": 0.01, "q11": 1, "r": 0.1, "p": 0.01}
        self.kalman = get_cst_speed_kalman(kalman_params)

        self.feedback_char_time = 0.7
        self.cur_pos = 0
        self.last_update_time = None

    def reset(self, pos, mes, zoom_fac):
        """
        pos in pos_bit
        mes in pixel
        zoom_fac in pos_bit / pixel
        """
        self.kalman.reset(np.array([[pos + mes * zoom_fac], [0]]))
        self.cur_pos = pos
        self.last_update_time = time.time()

    def update(self, pos, mes, zoom_fac):
        u = np.array([[0]])
        z = None if mes is None else np.array([[pos + mes * zoom_fac]])
        dt = time.time() - self.last_update_time
        self.kalman.F = np.array([[1, dt], [0, 1]])
        self.kalman.update(u, z)
        self.cur_pos = pos
        self.last_update_time = time.time()

    def get_target_speed(self):  # in px / sec
        pan_pos = self.kalman.x[0, 0]
        pan_speed = self.kalman.x[1, 0]
        return (pan_pos - self.cur_pos) / self.feedback_char_time + pan_speed


class CustomDriver(CameraDriver):
    def __init__(self):
        super().__init__()

        self.model = YOLO("runs/detect/train/weights/best.pt")

        self.add_trackbar("zoom", 11038, self.zoom_m, self.zoom_M)
        self.last_zoom_cmd = self.trackbar_data["zoom"]

        self.pan_driver = PosAxisDriver()
        self.tilt_driver = PosAxisDriver()

        self.moving_camera = False
        self.lost_for = 100000000
        self.max_lost_for = 10

        # self.save_frames = True
        self.tmp_path = "results/frames/cur_frame.jpg"

    def update_estimation(self, results, keypressed):
        # TODO : change that with the measured zoom
        zoom_fac = self.movement_model.zoom_factor(self.last_zoom_cmd)

        if len(results[0].boxes.xywh) == 0:
            self.lost_for += 1
            self.pan_driver.update(self.cur_pan, None, zoom_fac)
            self.tilt_driver.update(self.cur_tilt, None, zoom_fac)
        else:
            x, y, w, h = results[0].boxes.xywh[0].detach().cpu().numpy()
            if self.lost_for > self.max_lost_for:
                self.pan_driver.reset(self.cur_pan, x - 640 // 2, zoom_fac)
                self.tilt_driver.reset(self.cur_tilt, -y + 480 // 2, zoom_fac)
            else:
                self.pan_driver.update(self.cur_pan, x - 640 // 2, zoom_fac)
                self.tilt_driver.update(self.cur_tilt, -y + 480 // 2, zoom_fac)
            self.lost_for = 0

    def frame_callback(self, frame, keypressed):

        # setting camera target
        if self.frame_id == 0:
            camera_driver.send_pos_cmd([39, -70], [0x20, 0x20])
        elif self.frame_id == 1 or self.last_zoom_cmd != self.trackbar_data["zoom"]:
            self.last_zoom_cmd = self.trackbar_data["zoom"]
            self.send_zoom_cmd(self.last_zoom_cmd)

        self.send_pos_inq_cmd()
        self.update_read_buffer()

        pil_img = Image.fromarray(frame)
        pil_img.save(self.tmp_path)
        # save_load_img = np.array(Image.open(self.tmp_path))

        results = self.model.track(self.tmp_path, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        self.qt_handle.imshow("YOLOv8 Tracking", annotated_frame)

        # update the kalman estimation
        self.update_estimation(results, keypressed)

        # driving the camera
        if keypressed == Qt.Key_T:
            self.moving_camera = True
        if self.lost_for > self.max_lost_for:
            self.moving_camera = False

        debug_text = []

        if self.moving_camera:
            target_pan_speed = self.pan_driver.get_target_speed()
            target_tilt_speed = self.tilt_driver.get_target_speed()
            pan_speed_cmd = self.movement_model.bit_speed_to_cmd(target_pan_speed)
            tilt_speed_cmd = self.movement_model.bit_speed_to_cmd(target_tilt_speed)
            debug_text.append("following")
            # print(self.tilt_driver.kalman.x.flatten(), tilt_speed_cmd)
            # if len(results[0].boxes.xywh) > 0:
            #     self.all_screen_pos.append(
            #         results[0].boxes.xywh[0].detach().cpu().numpy()[0]
            #     )
            # else:
            #     self.all_screen_pos.append(None)
            # self.all_pan_pos.append(self.cur_pan)
        else:
            pan_speed_cmd = 0
            tilt_speed_cmd = 0
            debug_text.append("stoped")
            # print("not moving the cam", self.lost_for)

        self.send_speed_cmd([pan_speed_cmd, -tilt_speed_cmd])

        # keypress handling
        if keypressed == Qt.Key_Space:
            print("spacebar has been pressed")
            self.save_frames = not self.save_frames

        if self.save_frames:
            debug_text.append("saving frames")

        print(" ".join(debug_text))

        return frame, True


if __name__ == "__main__":
    camera_driver = CustomDriver()
    camera_driver.main_loop()

    """
    ffmpeg -framerate 25 -pattern_type glob -i 'results/frames/*.jpg' -c:v libx264 -pix_fmt yuv420p results/videos/out.mp4
    rm results/frames/*
    """
