from ultralytics import YOLO
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import tqdm

"""
python -m yolo.train_yolo_model

Permet d'entraîner un modèle YOLO de Ultralics. 
/!\ Ne pas oublier de bien spécifier les dataset qu'on veut utiliser (ligne 137)

Le scan à la main que je fais des dataset est assez long pour coco. Ne pas hésiter à le débrancher si besoins au début.
"""

test_id = 0

np.random.seed(42)


def rand_modify_image(img_src_path, img_tgt_path):
    img = Image.open(img_src_path)

    if np.random.random() < 0.8:
        blur_level = np.random.uniform(0.2, 2)
        img = img.filter(ImageFilter.GaussianBlur(blur_level))

    quality = int(np.random.choice([95, 70, 50, 40, 30, 20]))
    img.save(img_tgt_path, quality=quality)


class SourceDataset:
    def __init__(
        self,
        root_path,
        target_path,
        rand_aug=False,
        label_exceptions=[],
        limit=None,
        copy_labels=True,
    ):
        self.root_path = root_path
        self.target_path = target_path
        self.rand_aug = rand_aug
        self.label_exceptions = label_exceptions
        self.limit = limit
        self.copy_labels = copy_labels

        self.name = self.root_path.name

        self.scan_available_imgs()

    def scan_available_imgs(self):
        print("Scanning images from dataset", self.name)
        self.all_train_stems = self.scan_folder("train")
        self.all_val_stems = self.scan_folder("val")

    def extract_label_ids(self, label_path):
        with open(label_path) as f:
            data = f.readlines()
        ids = []
        for row in data:
            splitted = row.split(" ")
            if len(splitted) == 5:
                ids.append(int(splitted[0]))
        return ids

    def scan_folder(self, folder):
        available_stems = []
        img_folder_path = self.root_path / "images" / folder
        lab_folder_path = self.root_path / "labels" / folder
        for img_path in tqdm.tqdm(list(img_folder_path.glob("*"))):
            label_path = lab_folder_path / (img_path.stem + ".txt")
            if label_path.exists():
                cur_ids = set(self.extract_label_ids(label_path))
                if len(cur_ids.intersection(self.label_exceptions)) == 0:
                    available_stems.append((img_path.name, img_path.stem))
            else:
                available_stems.append((img_path.name, img_path.stem))

        if self.limit is not None and len(available_stems) > self.limit:
            chosen = np.random.choice(
                len(available_stems), size=self.limit, replace=False
            )
            available_stems = [available_stems[i] for i in chosen]

        return available_stems

    def send_imgs(self):
        print("Sending images from dataset", self.name)
        self.send_folder("train", self.copy_labels, self.all_train_stems)
        self.send_folder("val", self.copy_labels, self.all_val_stems)

    def re_send_train_imgs(self):
        if self.rand_aug:
            print("Re-Sending images from dataset", self.name)
            self.send_folder("train", self.copy_labels, self.all_train_stems)

    def send_folder(self, folder, send_labels, stems):

        for name, stem in tqdm.tqdm(stems):
            src_img_path = self.root_path / "images" / folder / name
            tgt_img_path = self.target_path / "images" / folder / (self.name + name)
            src_label_path = self.root_path / "labels" / folder / (stem + ".txt")
            tgt_label_path = (
                self.target_path / "labels" / folder / (self.name + stem + ".txt")
            )

            # TODO : add image rand augment
            if self.rand_aug:
                rand_modify_image(src_img_path, tgt_img_path)
            else:
                shutil.copy(src_img_path, tgt_img_path)
            if send_labels and src_label_path.exists():
                shutil.copy(src_label_path, tgt_label_path)


def remake_target_dataset():
    if tp.exists():
        shutil.rmtree(tp)
    (tp / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tp / "images" / "val").mkdir(parents=True, exist_ok=True)
    (tp / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tp / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ds.send_imgs()


def update_target_dataset(*args, **kwargs):
    for ds in datasets:
        ds.re_send_train_imgs()


# Load a model
if False:

    sp = Path("/media/oscar/Data/datasets/sources")
    tp = Path("results/dataset/composite_train")
    datasets = [
        SourceDataset(
            sp / "coco",
            tp,
            rand_aug=False,
            label_exceptions=[4],
            limit=1000,
            copy_labels=False,
        ),
        SourceDataset(sp / "blender", tp, rand_aug=True),
        SourceDataset(sp / "room_hanging_1", tp, rand_aug=False),
        SourceDataset(sp / "room_hanging_2", tp, rand_aug=False),
        SourceDataset(sp / "room_hanging_3", tp, rand_aug=False),
        SourceDataset(sp / "room_hanging_4", tp, rand_aug=False),
    ]

    remake_target_dataset()
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model.add_callback("on_train_epoch_end", update_target_dataset)

    # Train the model
    try:
        results = model.train(
            data="yolo/yaml/solo.yaml",
            # data="coco8.yaml",
            epochs=50,
            imgsz=640,
            plots=True,
            # mosaic=0.1,
            mixup=0.05,
            copy_paste=0.1,
            crop_fraction=0.1,
            hsv_h=0.3,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
else:
    model = YOLO("runs/detect/train9/weights/best.pt")


def test_model():
    global test_id

    # run evaluation on the test images
    test_path = Path("/home/oscar/workspace/plane_follow/python/yolo/test_imgs/imgs")
    test_save_path = Path(
        "/home/oscar/workspace/plane_follow/python/yolo/test_imgs/prediction"
    )
    test_save_path.mkdir(exist_ok=True, parents=True)
    imgs_paths = list(test_path.glob("*"))
    results = model(imgs_paths)
    for img_path, result in zip(imgs_paths, results):
        result.save(str(test_save_path / img_path.name))

    test_id += 1


test_model()

# get the coco dataset
if False:
    from ultralytics.utils.downloads import download
    from pathlib import Path

    # Download labels
    segments = False  # segment or box labels
    dir = Path("/media/oscar/Data/datasets/coco")  # dataset root dir
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
    urls = [
        url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")
    ]  # labels
    download(urls, dir=dir.parent)
    # Download data
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
        "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
        # "http://images.cocodataset.org/zips/test2017.zip",
    ]  # 7G, 41k images (optional)
    download(urls, dir=dir / "images", threads=3)
