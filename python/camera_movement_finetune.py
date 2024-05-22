import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt

from camera_commons import CameraDriver

"""
python camera_movement_finetune.py

Fait bouger la caméra à différentes vitesse pour mesurer la relation entre la commande de vitesse et la vitesse effective mesurée par la caméra.
"""


class Experiment:
    def __init__(self, zoom, speed, hold_time):
        self.zoom = zoom
        self.speed = speed
        self.hold_time = hold_time

        self.all_times = []
        self.all_pos = []

    def update(self, time, pos):
        self.all_times.append(time)
        self.all_pos.append(pos)

    def start(self, frame_id):
        self.start_frame = frame_id

    def cur_speed_cmd(self, frame_id):
        if frame_id == self.start_frame:
            return self.speed
        elif frame_id == self.start_frame + self.hold_time:
            return 0
        else:
            return None

    def is_done(self, frame_id):
        return frame_id > self.start_frame + self.hold_time * 2


class CustomDriver(CameraDriver):
    def __init__(self, exp_name):
        super().__init__()

        self.exp_name = exp_name
        # self.using_plt = True

        # to test cmd speed
        if exp_name == "tilt" or exp_name == "pan":
            self.all_exps = [Experiment(11190, i, 25) for i in range(1, 30, 1)] + [
                Experiment(2000, i, 25) for i in range(1, 30, 1)
            ]
        elif exp_name == "zoom":
            # to test the conversion between px speed and cmd even in the presence of zoom
            self.all_exps = [
                Experiment(11190, self.movement_model.px_speed_to_cmd(480, 11190), 25)
            ]
        else:
            raise ValueError("only 'pan', 'tilt' are accepted as exp_name")

        self.cur_exp_id = -1
        self.cur_exp = None
        self.test_speed = 0
        self.start_frame = None

    def frame_callback(self, frame, keypressed):

        if self.cur_exp_id >= len(self.all_exps):
            return frame, False

        if self.cur_exp_id == -1:
            if self.frame_id % 2 == 0:
                self.send_zoom_cmd(self.all_exps[0].zoom)
            else:
                self.send_pos_cmd([-2000, 0], [30, 30])

        # keypress handling
        if keypressed == Qt.Key_Space or (
            self.start_frame is not None
            and (self.frame_id - self.start_frame) == 25 * 4
        ):
            self.start_frame = self.frame_id
            print("Starting next experiment")
            self.cur_exp_id = self.cur_exp_id + 1
            self.cur_exp = self.all_exps[self.cur_exp_id]

            self.cur_exp.start(self.frame_id)

        if self.cur_exp is not None:
            if not self.cur_exp.is_done(self.frame_id):
                if self.exp_name == "pan":
                    self.cur_exp.update(self.pos_update_time, self.cur_pan)
                if self.exp_name == "tilt":
                    self.cur_exp.update(self.pos_update_time, self.cur_tilt)

            cur_speed_cmd = self.cur_exp.cur_speed_cmd(self.frame_id)
            if cur_speed_cmd is not None:
                if self.exp_name == "pan":
                    self.send_speed_cmd([cur_speed_cmd, 0])  # max 24
                if self.exp_name == "tilt":
                    self.send_speed_cmd([0, -cur_speed_cmd])  # max 24

            if self.cur_exp.is_done(self.frame_id):
                self.cur_exp = None
                self.send_pos_cmd([-2000, 0], [30, 30])
                time.sleep(0.1)
                if self.cur_exp_id + 1 < len(self.all_exps):
                    print("cur experiment is done")
                    self.send_zoom_cmd(self.all_exps[self.cur_exp_id + 1].zoom)

        self.send_pos_inq_cmd()
        self.update_read_buffer()

        return frame, True

    def plot_exp(self, exp: Experiment):
        t = np.array(exp.all_times)
        t = t - t[0]
        plt.plot(t, exp.all_pos)


if __name__ == "__main__":

    exp_name = "tilt"

    if exp_name not in ["pan", "tilt"]:
        raise ValueError("only 'pan', 'tilt' are accepted as exp_name")

    if False:
        camera_driver = CustomDriver(exp_name)
        camera_driver.main_loop()
        with open(f"results/camera_movement/exps_{exp_name}.pkl", "wb") as f:
            pkl.dump(camera_driver.all_exps, f)
    else:
        with open(f"results/camera_movement/exps_{exp_name}.pkl", "rb") as f:
            exps = pkl.load(f)

        for exp in exps:
            exp.all_times = np.array(exp.all_times)
            exp.all_pos = np.array(exp.all_pos)
            exp.all_times = exp.all_times - exp.all_times[0]

        plt.figure()
        for exp in exps:
            label_txt = "zoom {}  |  speed {}"
            plt.plot(
                exp.all_times, exp.all_pos, label=label_txt.format(exp.zoom, exp.speed)
            )

        all_cmd = []
        all_speed = []

        plt.figure()
        for exp in exps:
            label_txt = "zoom {}  |  speed {}"
            pos_delta = exp.all_pos[1:] - exp.all_pos[:-1]
            time_delta = exp.all_times[1:] - exp.all_times[:-1]
            time_mid = (exp.all_times[1:] + exp.all_times[:-1]) / 2
            speed = pos_delta / time_delta
            plt.plot(
                time_mid,
                speed,
                label=label_txt.format(exp.zoom, exp.speed),
            )
            xlim = plt.xlim()
            mean_speed = np.mean(speed[8:25])
            plt.plot(xlim, [mean_speed, mean_speed], "--k")
            plt.xlim(xlim)
            all_cmd.append(exp.speed)
            all_speed.append(mean_speed)
        plt.legend()

        plt.figure()
        plt.plot(all_cmd, all_speed, "o")
        all_cmd = np.array(all_cmd)
        all_speed = np.array(all_speed)

        low_speed_id = np.where(all_cmd < 4)
        x = plt.xlim()
        x = np.array([0, x[1]])

        a = all_cmd[low_speed_id]
        a = np.stack([a, np.ones_like(a)], axis=1)
        b = all_speed[low_speed_id].reshape((-1, 1))

        low_fac = np.linalg.lstsq(a, b)[0][:, 0]
        plt.plot(x, x * low_fac[0] + low_fac[1], "--k")

        high_speed_id = np.where(np.logical_and(4 < all_cmd, all_cmd <= 12))
        x = plt.xlim()
        x = np.array([0, x[1]])

        a = all_cmd[high_speed_id]
        a = np.stack([a, np.ones_like(a)], axis=1)
        b = all_speed[high_speed_id].reshape((-1, 1))

        high_fac = np.linalg.lstsq(a, b)[0][:, 0]
        plt.plot(x, x * high_fac[0] + high_fac[1], "--k")

        mid_speed_id = np.where(4 == all_cmd)
        mid_speed = np.mean(all_speed[mid_speed_id])

        plt.grid()

        vel_cmd_translate = []
        for cmd in range(1, 4):
            vel_cmd_translate.append((cmd, cmd * low_fac[0] + low_fac[1]))
        vel_cmd_translate.append((4, mid_speed))
        for cmd in range(5, 18):
            vel_cmd_translate.append((cmd, cmd * high_fac[0] + high_fac[1]))

        inv_cmd_translate = []
        for cmd, speed in vel_cmd_translate[::-1]:
            inv_cmd_translate.append((-cmd, -speed))

        full_cmd_to_vel = inv_cmd_translate + [(0, 0)] + vel_cmd_translate

        print(full_cmd_to_vel)

        plt.show()
