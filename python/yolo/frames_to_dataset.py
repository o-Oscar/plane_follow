import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PyQt5.QtCore import QEvent, QObject, Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
import tqdm
from ultralytics import YOLO

"""
python -m yolo.frames_to_dataset --tgt /media/oscar/Data/datasets/sources/room_hanging_0 --src results/frames/exp_0000 --network runs/detect/train/weights/best.pt
python -m yolo.frames_to_dataset --tgt /media/oscar/Data/datasets/sources/room_hanging_0 --network runs/detect/train/weights/best.pt
python -m yolo.frames_to_dataset --tgt /media/oscar/Data/datasets/sources/room_hanging_0
python -m yolo.frames_to_dataset --tgt /media/oscar/Data/datasets/sources/room_hanging_3 --src results/frames/exp_0003 --network runs/detect/train7/weights/best.pt
python -m yolo.frames_to_dataset --tgt /media/oscar/Data/datasets/sources/room_hanging_4 --src results/frames/exp_0004 --network runs/detect/train8/weights/best.pt

runs/detect/train8/weights/best.pt


Utiliser ce script pour transformer un ensemble de frames à plat (typiquement récupéré par du tracking) en un dataest interprétable par le YOLO de Ultralitics.
L'argument (optionel) --src permet de spécifier le chemin vers les frames à plat
L'argument (optionel) --network permet de spécifier le chemin vers le network à utiliser. Ce chemin peut être modifié dans la gui.
L'argument --tgt permet de spécifier le chemin vers le dataset yolo à modifier. 
"""


parser = argparse.ArgumentParser()
parser.add_argument("--tgt", type=str, help="target folder", required=True)
parser.add_argument("--src", type=str, help="source folder", default=None)
parser.add_argument(
    "--network", type=str, help="network used for initial labeling", default=None
)
args = parser.parse_args()


args.tgt = Path(args.tgt)

if args.src is not None:
    args.src = Path(args.src)

if args.network is not None:
    args.network = Path(args.network)

# If source is specified : create the tgt folder and copy the images
if args.src is not None:
    if args.tgt.exists():
        raise FileExistsError("You can not specify a target that already exists")

    # create the dataset structure
    (args.tgt / "images" / "train").mkdir(parents=True, exist_ok=True)
    (args.tgt / "images" / "val").mkdir(parents=True, exist_ok=True)
    (args.tgt / "images" / "unused").mkdir(parents=True, exist_ok=True)
    (args.tgt / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (args.tgt / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (args.tgt / "labels" / "unused").mkdir(parents=True, exist_ok=True)

    # copy the images and create an empty label file
    all_imgs = list(args.src.glob("*"))
    print("copying files")
    for img_src in tqdm.tqdm(all_imgs):
        img_name = img_src
        img_tgt = args.tgt / "images" / "unused" / img_src.name
        shutil.copy(img_src, img_tgt)

        label_tgt = args.tgt / "labels" / "unused" / (img_name.stem + ".txt")
        with open(label_tgt, "w") as f:
            pass


class State:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.folders = ["train", "val", "unused"]

        self.model = None
        self.cur_yolo_box = None

        self.read_cur_state()
        self.set_cur_img_id(0)

    def set_model(self, model_path):
        self.model = YOLO(model_path)
        self.compute_model_preds()

    # def paths_to_pred(self, paths):
    #     preds = self.model.predict(paths, verbose=False)
    #     for pred in preds:
    #         if len(pred.boxes.xywh) == 0:
    #             self.all_box_pred.append(None)
    #         else:
    #             self.all_box_pred.append(pred.boxes.xywh[0])

    # def compute_model_preds(self):
    #     all_img_paths = []
    #     self.all_box_pred = []
    #     for i, row in tqdm.tqdm(self.all_imgs.iterrows(), total=len(self.all_imgs)):
    #         img_path = Path(self.root_path / "images" / row["folder"] / row["name"])
    #         all_img_paths.append(img_path)
    #         if len(all_img_paths) == 100:
    #             self.paths_to_pred(all_img_paths)
    #             all_img_paths = []
    #     if len(all_img_paths) > 0:
    #         self.paths_to_pred(all_img_paths)

    def read_cur_state(self):
        raw_data = []
        for sub_dir in self.folders:
            cur_path = self.root_path / "images" / sub_dir
            for img_path in cur_path.glob("*"):
                raw_data.append([img_path.name, img_path.stem, sub_dir])

        self.all_imgs = pd.DataFrame(raw_data, columns=["name", "stem", "folder"])
        self.all_imgs = self.all_imgs.sort_values("name").reset_index(drop=True)

    def get_cur_img(self):
        name = self.all_imgs["name"].iloc[self.cur_img_id]
        folder = self.get_cur_folder()
        return Image.open(self.root_path / "images" / folder / name)

    def get_cur_folder(self):
        return self.all_imgs["folder"].iloc[self.cur_img_id]

    def move_image_to_folder(self, tgt_folder):
        src_folder = self.all_imgs["folder"].iloc[self.cur_img_id]
        img_name = self.all_imgs["name"].iloc[self.cur_img_id]
        img_stem = self.all_imgs["stem"].iloc[self.cur_img_id]

        print(
            "moving",
            self.root_path / "images" / src_folder / img_name,
            "to",
            self.root_path / "images" / tgt_folder / img_name,
        )
        shutil.move(
            self.root_path / "images" / src_folder / img_name,
            self.root_path / "images" / tgt_folder / img_name,
        )

        shutil.move(
            self.root_path / "labels" / src_folder / (img_stem + ".txt"),
            self.root_path / "labels" / tgt_folder / (img_stem + ".txt"),
        )

        self.all_imgs.loc[self.cur_img_id, "folder"] = tgt_folder

    def hide(self, src_folder):
        tgt_folder = "hide"
        (self.root_path / "images" / tgt_folder).mkdir(parents=True, exist_ok=True)
        (self.root_path / "labels" / tgt_folder).mkdir(parents=True, exist_ok=True)

        for i, row in self.all_imgs[self.all_imgs["folder"] == src_folder].iterrows():
            img_name = row["name"]
            img_stem = row["stem"]
            shutil.move(
                self.root_path / "images" / src_folder / img_name,
                self.root_path / "images" / tgt_folder / img_name,
            )

            shutil.move(
                self.root_path / "labels" / src_folder / (img_stem + ".txt"),
                self.root_path / "labels" / tgt_folder / (img_stem + ".txt"),
            )

        self.read_cur_state()
        self.set_cur_img_id(0)

    def set_cur_img_id(self, img_id):
        self.cur_img_id = min(max(img_id, 0), len(self.all_imgs) - 1)
        self.read_box_data()
        self.update_yolo_pred()

    def update_yolo_pred(self):
        src_folder = self.all_imgs["folder"].iloc[self.cur_img_id]
        img_name = self.all_imgs["name"].iloc[self.cur_img_id]
        image_path = self.root_path / "images" / src_folder / img_name

        if self.model is None:
            return
        pred = self.model.predict(image_path, verbose=False)[0]
        if len(pred.boxes.xywh) == 0:
            self.cur_yolo_box = None
        else:
            self.cur_yolo_box = pred.boxes.xywh[0]

    def read_box_data(self):
        src_folder = self.all_imgs["folder"].iloc[self.cur_img_id]
        img_stem = self.all_imgs["stem"].iloc[self.cur_img_id]
        label_path: Path = self.root_path / "labels" / src_folder / (img_stem + ".txt")

        if label_path.exists():
            with open(label_path) as f:
                data = f.readline()
            if len(data) > 0:
                _, x, y, w, h = map(float, data.split(" "))
                self.set_xywh(x, y, w, h)
                self.has_box = True
            else:
                self.has_box = False
        else:
            self.has_box = False

    def write_box_data(self):
        src_folder = self.all_imgs["folder"].iloc[self.cur_img_id]
        img_stem = self.all_imgs["stem"].iloc[self.cur_img_id]
        label_path: Path = self.root_path / "labels" / src_folder / (img_stem + ".txt")
        with open(label_path, "w") as f:
            f.write(" ".join(map(str, [0] + list(self.xywh()))))

    def remove_cur_label(self):
        src_folder = self.all_imgs["folder"].iloc[self.cur_img_id]
        img_stem = self.all_imgs["stem"].iloc[self.cur_img_id]
        label_path: Path = self.root_path / "labels" / src_folder / (img_stem + ".txt")
        with open(label_path, "w") as f:
            pass
        self.has_box = False

    def set_xywh(self, x, y, w, h):
        self.ulx = (x - w / 2) * 640
        self.uly = (y - h / 2) * 480
        self.brx = (x + w / 2) * 640
        self.bry = (y + h / 2) * 480

    def xywh(self):
        x = (self.ulx + self.brx) / 2 / 640
        y = (self.uly + self.bry) / 2 / 480
        w = (self.brx - self.ulx) / 640
        h = (self.bry - self.uly) / 480
        return x, y, w, h

    def set_ul(self, x, y):
        self.ulx = x
        self.uly = y
        if self.has_box is False:
            self.has_box = True
            self.brx = 640
            self.bry = 480
        self.write_box_data()

    def set_br(self, x, y):
        self.brx = x
        self.bry = y
        if self.has_box is False:
            self.has_box = True
            self.ulx = 0
            self.uly = 0
        self.write_box_data()

    def label_box(self):
        if not self.has_box:
            return None
        return self.ulx, self.uly, self.brx, self.bry

    def network_box(self):
        if self.model is None:
            return None
        return self.cur_yolo_box


class DatasetSlider(QSlider):
    def __init__(self, value_changed_callback):
        super().__init__(Qt.Orientation.Horizontal)
        self.valueChanged.connect(value_changed_callback)


class ImageLabel(QLabel):
    def __init__(self, left_click_callback, right_click_callback):
        super().__init__()

        self.left_click_callback = left_click_callback
        self.right_click_callback = right_click_callback

        self.setFixedSize(640, 480)

        self.setMouseTracking(True)
        self.installEventFilter(self)

        self.img = None
        self.label_box = None
        self.network_box = None
        self.label_color = Qt.GlobalColor.blue
        self.network_color = Qt.GlobalColor.red

        # QPen pen;
        # pen.setWidth(40);
        # pen.setColor(Qt::red);

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == 1:
                self.left_click_callback(event.pos().x(), event.pos().y())
            if event.button() == 2:
                self.right_click_callback(event.pos().x(), event.pos().y())
        if event.type() == QEvent.Type.MouseMove:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.left_click_callback(event.pos().x(), event.pos().y())
            if event.buttons() & Qt.MouseButton.RightButton:
                self.right_click_callback(event.pos().x(), event.pos().y())
        return False

    def set_img(self, img):
        self.img = img

    def set_label_box(self, label_box):  # xyxy
        self.label_box = label_box

    def set_network_box(self, network_box):  # xywh (centered)
        self.network_box = network_box

    def paintEvent(self, *args, **kwargs):
        painter = QPainter(self)
        if self.img is not None:
            qImg = QImage(np.array(self.img), 640, 480, 3 * 640, QImage.Format_RGB888)
            painter.drawImage(QRect(0, 0, 640, 480), qImg)
        if self.label_box is not None:
            qbox = self.xyxy_to_qbox(self.label_box)
            painter.setPen(self.label_color)
            painter.drawRect(*(int(x) for x in qbox))
        if self.network_box is not None:
            qbox = self.xywh_to_qbox(self.network_box)
            painter.setPen(self.network_color)
            painter.drawRect(*(int(x) for x in qbox))

    def xyxy_to_qbox(self, xyxy):
        return [
            int(x)
            for x in (
                xyxy[0],
                xyxy[1],
                xyxy[2] - xyxy[0],
                xyxy[3] - xyxy[1],
            )
        ]

    def xywh_to_qbox(self, xywh):
        x, y, w, h = xywh
        return [int(x) for x in (x - w / 2, y - h / 2, w, h)]


def get_spacer():
    return QSpacerItem(
        10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
    )


class MyRadioButton(QRadioButton):
    def __init__(self, name, key=None):
        super().__init__(name)
        self.name = name
        if key is None:
            self.key = name


class MainWindow(QMainWindow):
    def __init__(self, state: State):

        super().__init__()

        self.state = state

        self.setWindowTitle("label tool")
        self.setGeometry(100, 100, 1000, 800)

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        bot_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bot_layout)

        # img label
        img_container = QWidget()
        img_container_layout = QGridLayout()
        img_container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_container.setLayout(img_container_layout)
        self.img_label = ImageLabel(self.left_click_callback, self.right_click_callback)
        img_container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        img_container_layout.addItem(get_spacer(), 0, 0)
        img_container_layout.addWidget(self.img_label, 1, 1)
        img_container_layout.addItem(get_spacer(), 2, 2)

        top_layout.addWidget(img_container, 100)

        # tool boxes
        tool_boxes = QWidget()
        tool_boxes.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        )
        tool_box_layout = QVBoxLayout()
        tool_boxes.setLayout(tool_box_layout)
        top_layout.addWidget(tool_boxes, 0)

        # folder group
        folder_group_box = QGroupBox("Image Folder")
        folder_layout = QVBoxLayout()
        folder_group_box.setLayout(folder_layout)

        self.folder_buttons = {}
        for k in state.folders:
            self.folder_buttons[k] = QRadioButton(k)
            self.folder_buttons[k].toggled.connect(
                lambda x, k=k: self.toggled(k) if x else None
            )
            folder_layout.addWidget(self.folder_buttons[k])

        tool_box_layout.addWidget(folder_group_box, 0)

        # network group
        network_group_box = QGroupBox("Yolo Network")
        network_layout = QVBoxLayout()
        network_group_box.setLayout(network_layout)

        self.line_edit = QLineEdit()
        self.line_edit.textChanged.connect(self.network_path_changed)
        accept_current_button = QPushButton("Accept yolo as label (current)")
        accept_all_button = QPushButton("Accept yolo as label (all imgs)")
        network_layout.addWidget(self.line_edit)
        network_layout.addWidget(accept_current_button)
        network_layout.addWidget(accept_all_button)

        tool_box_layout.addWidget(network_group_box, 0)

        # misc group
        misc_group_box = QGroupBox("misc")
        misc_layout = QVBoxLayout()
        misc_group_box.setLayout(misc_layout)

        remove_box_button = QPushButton("remove label (r)")
        remove_box_button.clicked.connect(self.remove_cur_label)
        misc_layout.addWidget(remove_box_button)

        hide_unused_button = QPushButton("hide unused (h)")
        hide_unused_button.clicked.connect(self.hide_unused)
        misc_layout.addWidget(hide_unused_button)
        tool_box_layout.addWidget(misc_group_box, 0)

        tool_box_layout.addWidget(QWidget(), 100)

        # bottom row
        self.dataset_slider = DatasetSlider(self.slider_value_changed)
        self.dataset_label = QLabel("({:06d}/{:06d})".format(0, 0))

        bot_layout.addWidget(self.dataset_slider)
        bot_layout.addWidget(self.dataset_label)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # misc
        self.installEventFilter(self)

        if args.network is not None:
            self.line_edit.setText(str(args.network))
        self.update_all(force=True)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Q:
                self.close()
            if event.key() == Qt.Key.Key_R:
                self.remove_cur_label()
            if event.key() == Qt.Key.Key_Left:
                self.state.set_cur_img_id(self.state.cur_img_id - 1)
                self.update_all()
            if event.key() == Qt.Key.Key_Right:
                self.state.set_cur_img_id(self.state.cur_img_id + 1)
                self.update_all()

            if event.key() == Qt.Key.Key_A:
                self.folder_buttons["train"].setChecked(True)
            if event.key() == Qt.Key.Key_Z:
                self.folder_buttons["val"].setChecked(True)
            if event.key() == Qt.Key.Key_E:
                self.folder_buttons["unused"].setChecked(True)
        return False

    def network_path_changed(self):
        # runs/detect/train/weights/best.pt
        network_path = Path(self.line_edit.text())
        if network_path.exists() and network_path.suffix == ".pt":
            state.set_model(network_path)

        self.update_network_box(True)

    def remove_cur_label(self):
        self.state.remove_cur_label()
        self.update_label_box()

    def toggled(self, name):
        if name != self.state.get_cur_folder():
            print(name, self.state.get_cur_folder())
            self.state.move_image_to_folder(name)

    def slider_value_changed(self):
        self.state.set_cur_img_id(self.dataset_slider.value())
        self.update_all()

    def left_click_callback(self, x, y):
        self.state.set_ul(x, y)
        self.update_label_box()

    def right_click_callback(self, x, y):
        self.state.set_br(x, y)
        self.update_label_box()

    def hide_unused(self):
        self.state.hide("unused")
        self.update_all()

    def update_all(self, *, force=False):
        self.update_slider(force)
        self.update_image(force)
        self.update_folder_box(force)
        self.update_label_box(force)
        self.update_network_box(force)
        self.update_slider_text(force)

    def update_slider(self, force):
        if force or self.dataset_slider.value() != self.state.cur_img_id:
            self.dataset_slider.setValue(self.state.cur_img_id)
            self.dataset_slider.setMaximum(len(self.state.all_imgs) - 1)

    def update_slider_text(self, force):
        self.dataset_label.setText(
            "({:06d}/{:06d})".format(self.state.cur_img_id, len(self.state.all_imgs))
        )

    def update_image(self, force):
        img = self.state.get_cur_img()
        self.img_label.set_img(img)
        self.img_label.repaint()
        # self.img_label.setPixmap(QPixmap.fromImage(qImg))
        # self.img_label.set

    def update_folder_box(self, force):
        if force or not self.folder_buttons[self.state.get_cur_folder()].isChecked():
            for k, box in self.folder_buttons.items():
                box.setChecked(False)
            self.folder_buttons[self.state.get_cur_folder()].setChecked(True)

    def update_label_box(self, force=False):
        self.img_label.set_label_box(self.state.label_box())
        self.img_label.repaint()

    def update_network_box(self, force=False):
        self.img_label.set_network_box(self.state.network_box())
        self.img_label.repaint()


if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    state = State(args.tgt)
    main_window = MainWindow(state)
    main_window.show()
    main_window.showMaximized()

    app.exec()
