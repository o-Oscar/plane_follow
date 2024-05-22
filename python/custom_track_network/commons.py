import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, Subset

from custom_track_network.plotting import plot_validation
from scipy.ndimage import affine_transform
import kornia


def to_proper_rotmat(output):
    pred = output.reshape((-1, 3, 3))
    u, s, vh = np.linalg.svd(pred)
    proper_rotmat = np.einsum("bij,bjk->bik", u, vh)
    return proper_rotmat


def rotmat_error(target, output):
    pred_R = Rotation.from_matrix(output)
    target_R = Rotation.from_matrix(target)

    diff_R = pred_R * target_R.inv()
    rots = diff_R.as_rotvec()
    rots_mag = np.sqrt(np.sum(np.square(rots), axis=1))
    return rots_mag


def rot_pred_error(target, output):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(output) == torch.Tensor:
        output = output.detach().cpu().numpy()

    if target.shape[1] == 12:
        target = target[:, :9]
    if output.shape[1] == 12:
        output = output[:, :9]

    target = target.reshape((-1, 3, 3))
    output = output.reshape((-1, 3, 3))
    return rotmat_error(target, to_proper_rotmat(output))


def position_pred_error(target, output):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(output) == torch.Tensor:
        output = output.detach().cpu().numpy()

    if target.shape[1] == 12:
        target = target[:, 9:11]
    if output.shape[1] == 12:
        output = output[:, 9:11]

    return np.sqrt(np.sum(np.square(target - output), axis=1))


def scale_pred_error(target, output):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(output) == torch.Tensor:
        output = output.detach().cpu().numpy()

    if target.shape[1] == 12:
        target = target[:, 11:12]
    if output.shape[1] == 12:
        output = output[:, 11:12]

    return np.abs(target - output)


def all_pred_error(target, output):
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(output) == torch.Tensor:
        output = output.detach().cpu().numpy()

    return (
        rot_pred_error(target, output),
        position_pred_error(target, output),
        scale_pred_error(target, output),
    )


def create_model(output_size):
    model = timm.create_model("mobilenetv2_100", pretrained=True)
    model.reset_classifier(output_size)

    config = resolve_data_config({}, model=model)
    config["crop_pct"] = 0.5
    transform = create_transform(**config)

    return model, config, transform


class OrientationDataset_old(Dataset):
    def __init__(self, root_dir, config):
        self.quat_df = pd.read_csv(root_dir / "labels.csv")
        self.root_dir = root_dir
        self.config = config
        self.img_name_fmt = "{:04d}.jpeg"

        xyzw_quat = self.quat_df[["x", "y", "z", "w"]].to_numpy()
        rots = Rotation.from_quat(xyzw_quat)
        self.mats = rots.inv().as_matrix().reshape((-1, 9))

        self.img_mean = np.array(self.config["mean"]).reshape([3, 1, 1])
        self.img_std = np.array(self.config["std"]).reshape([3, 1, 1])

    def __len__(self):
        # return 10
        return len(self.quat_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        offset_params = np.random.uniform(-1, 1, size=3)

        img_name = self.root_dir / "imgs" / self.img_name_fmt.format(idx)
        image = np.array(Image.open(img_name)).transpose((2, 0, 1)) / 255  # 570, 320, 3
        image = crop_and_scale(image, self.config["input_size"], offset_params)
        image = (image - self.img_mean) / self.img_std
        image = image.astype(np.float32)

        rot_mat = self.mats[idx]
        target = np.concatenate([rot_mat, offset_params]).astype(np.float32)
        sample = (image, target)

        return sample


def crop_and_scale(image, target_size, offset_param):
    s0 = image.shape[1]
    s1 = target_size[1]

    off0 = np.ones(2) * s0 / 2
    off1 = np.ones(2) * s1 / 2 + offset_param[:2] * 50

    a = 0  # zero for now. Otherwise we need to compute the rotation for the target as well

    zoom_coef = 1.6 + 0.4 * offset_param[-1]
    rot_mat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * zoom_coef

    return np.stack(
        [
            affine_transform(
                image[i], rot_mat, off0 - rot_mat @ off1, (s1, s1), np.float32
            )
            for i in range(3)
        ]
    )


class OrientationDataset(Dataset):
    def __init__(self, root_dir):
        self.quat_df = pd.read_csv(root_dir / "labels.csv")
        self.root_dir = root_dir
        self.img_name_fmt = "{:04d}.jpeg"

        xyzw_quat = self.quat_df[["x", "y", "z", "w"]].to_numpy()
        rots = Rotation.from_quat(xyzw_quat)
        self.mats = rots.inv().as_matrix().reshape((-1, 9))

        self.transform = transforms.ToTensor()

    def __len__(self):
        # return 10
        return len(self.quat_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / "imgs" / self.img_name_fmt.format(idx)
        image = self.transform(Image.open(img_name))
        rot_mat = self.mats[idx]
        sample = (image, rot_mat)

        return sample


class OrientationDataloader:
    def __init__(self, root_dir, config, device=torch.device("cpu"), no_csv=False):
        self.batch_size = 10
        self.s0 = 720
        self.s1 = 224

        self.config = config
        self.device = device
        self.no_csv = no_csv

        self.img_mean = np.array(self.config["mean"]).reshape([1, 3, 1, 1])
        self.img_std = np.array(self.config["std"]).reshape([1, 3, 1, 1])
        self.img_mean = torch.tensor(self.img_mean, dtype=torch.float32).to(self.device)
        self.img_std = torch.tensor(self.img_std, dtype=torch.float32).to(self.device)
        dataset = OrientationDataset(root_dir)

        self.base_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=None,
        )

    def __iter__(self):
        for imgs, targets in self.base_loader:

            imgs = imgs.to(self.device)

            M, offset_param = self.generate_random_crop_pos(imgs.shape[0])
            M = torch.tensor(M, dtype=torch.float32).to(self.device)

            if imgs.shape[-1] != 720 or imgs.shape[-2] != 720:
                print("coucou")
                imgs = kornia.geometry.transform.resize(imgs, (720, 720))
                test = imgs.detach().cpu().numpy()
                import matplotlib.pyplot as plt

                plt.imshow(test[0].transpose((1, 2, 0)))
                plt.show()

            imgs = kornia.geometry.transform.warp_affine(imgs, M, (self.s1, self.s1))
            imgs = (imgs - self.img_mean) / self.img_std

            offset_param = torch.tensor(offset_param)
            targets = torch.concat([targets, offset_param], dim=1).type(torch.float32)
            targets = targets.to(self.device)

            yield imgs, targets

    def generate_random_crop_pos(self, batch_size):
        s0 = self.s0
        s1 = self.s1

        width = np.random.uniform(s0 / 3, s0, batch_size)
        min_delta = np.maximum(0, 2 * s0 / 3 - width)
        max_delta = np.minimum(s0 / 3, s0 - width)
        dx = np.random.uniform(min_delta, max_delta)
        dy = np.random.uniform(min_delta, max_delta)

        scale = s1 / width
        pixel_dx = -dx * scale
        pixel_dy = -dy * scale
        zero = np.zeros(batch_size)
        M = np.array([[scale, zero, pixel_dx], [zero, scale, pixel_dy]])
        M = M.transpose((2, 0, 1))

        cposx = (s0 / 2 - dx) * s1 / width
        cposy = (s0 / 2 - dy) * s1 / width

        offx = cposx * 2 / s1 - 1
        offy = cposy * 2 / s1 - 1
        offz = width * 3 / s0 - 2

        offset_params = np.stack([offx, offy, offz], axis=1)

        return M, offset_params

    def __len__(self):
        return len(self.base_loader)
