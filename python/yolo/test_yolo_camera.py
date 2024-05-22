import cv2
from ultralytics import YOLO
from camera_commons import CameraDriver

import numpy as np
import cv2
from PyQt5.QtCore import Qt

"""
python -m yolo.test_yolo_camera

Script de base qui permet de vérifier si on a bien une communication avec la caméra à 25+ fps et une gui qui fonctionne.
"""


class CustomDriver(CameraDriver):
    def __init__(self):
        super().__init__()

        # model = YOLO("yolov8n.pt")
        # model = YOLO("runs/detect/train2/weights/best.pt")
        self.model = YOLO("runs/detect/train/weights/best.pt")

        # example of slider callback
        self.add_trackbar("pan", 39, self.pan_m, self.pan_M)
        self.add_trackbar("tilt", -70, self.tilt_m, self.tilt_M)
        self.add_trackbar("zoom", 11190, self.zoom_m, self.zoom_M)

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

        # results = self.model.track(frame, persist=True, verbose=False)
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        self.qt_handle.imshow("YOLOv8 Tracking", annotated_frame)

        if keypressed == Qt.Key_Space:
            print("spacebar has been pressed")

        return frame, True


if __name__ == "__main__":
    camera_driver = CustomDriver()
    camera_driver.main_loop()
