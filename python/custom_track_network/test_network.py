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
from custom_track_network.commons import (
    OrientationDataloader,
    create_model,
    plot_validation,
)

if __name__ == "__main__":
    model, config, transform = create_model(12)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.device = device
    model = model.to(device)

    checkpoint = torch.load("checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

    test_path = Path("/home/oscar/workspace/plane_follow/blender/test0/dataset/test")
    test_loader = OrientationDataloader(test_path, config, device)

    model.eval()

    with torch.no_grad():

        all_imgs = []
        all_targets = []
        all_outputs = []

        for i, (images, target) in enumerate(test_loader):

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)

            all_imgs.append(images.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())

    all_imgs = np.concatenate(all_imgs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    plot_validation(
        all_imgs,
        all_targets,
        all_outputs,
        config,
        show_targets=False,
        path="results/imgs/test",
    )
