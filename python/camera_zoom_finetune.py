import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt

from camera_commons import CameraDriver

"""
python -m camera_zoom_finetune

Script pour mesurer les carractéristiques de zoom et de résolution de la caméra.
Fait zoomer et bouger la caméra vers différentes positions. En cliquant sur l'écran, on indique un point statique dans le repère monde.
En utilisant l'information de la position de ce point sur l'écran, on en déduit les carractéristiques de zoom et de résolution de la caméra. 
"""


class CustomDriver(CameraDriver):
    def __init__(self):
        super().__init__()

        # self.all_zoom = [0, 4000, 8000, 12000, 16000]
        self.all_zoom = [1000 * i for i in range(0, 17)]

        self.start_pos = True
        self.cur_zoom_id = 0
        self.left = True

        self.all_xs = []
        self.cur_zoom_xs = []
        self.all_pan = []
        self.cur_pan = []

    def click_callback(self, x, y):
        if self.start_pos:
            self.start_pos = False
            return
        self.cur_zoom_xs.append(x)
        self.cur_pan.append(self.compute_pan())
        if self.left:
            self.left = False
            return
        else:
            self.left = True
            self.cur_zoom_id += 1
            self.all_xs.append(self.cur_zoom_xs)
            self.all_pan.append(self.cur_pan)
            self.cur_zoom_xs = []
            self.cur_pan = []

            if self.cur_zoom_id == len(self.all_zoom):
                with open("results/camera_movement/zoom_finetune.pkl", "wb") as f:
                    pkl.dump({"clicks": self.all_xs, "pbit_pos": self.all_pan}, f)
            return

    def frame_callback(self, frame, keypressed):
        if self.cur_zoom_id == len(self.all_zoom):
            return frame, False

        if self.start_pos:
            if self.frame_id % 2 == 0:
                self.send_pos_cmd([0, 0], [10, 10])
            else:
                self.send_zoom_cmd(self.all_zoom[-1])
        else:
            if self.frame_id % 2 == 0:
                self.send_pos_cmd([self.compute_pan(), 0], [10, 10])
            else:
                self.send_zoom_cmd(self.all_zoom[self.cur_zoom_id])

        return frame, True

    def compute_pan(self):
        zoom = self.all_zoom[self.cur_zoom_id]
        to_return = int(640 // 4 * self.movement_model.zoom_factor(zoom))
        return to_return * (1 if self.left else -1)


if __name__ == "__main__":
    if False:
        print(
            "click on a filxed point (in world space) on the screen after each camera movement"
        )
        camera_driver = CustomDriver()
        camera_driver.main_loop()
    else:
        with open("results/camera_movement/zoom_finetune.pkl", "rb") as f:
            data = pkl.load(f)

        all_pan = np.array(data["pbit_pos"])
        all_xs = np.array([data["clicks"]]).reshape((-1, 2))

        print(all_xs.shape)

        pan_diff = all_pan[:, 1] - all_pan[:, 0]
        xs_diff = all_xs[:, 1] - all_xs[:, 0]

        zoom_facs = -pan_diff / xs_diff
        zoom_cmds = np.array([1000 * i for i in range(0, 17)])

        plt.figure()
        # plt.plot(zoom_cmds, np.log(-zoom_facs))
        plt.plot(zoom_cmds, np.log(zoom_facs))

        zoom_coefs = np.polyfit(zoom_cmds, np.log(zoom_facs), 2)
        log_zoom_pred = 0
        zoom_cmds = np.array([1000 * i for i in range(0, 20)])
        for i, c in enumerate(zoom_coefs[::-1]):
            log_zoom_pred += c * zoom_cmds**i

        print(list(zoom_coefs))
        plt.plot(zoom_cmds, log_zoom_pred)

        plt.show()
