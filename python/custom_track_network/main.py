"""
base from https://github.com/pytorch/examples/blob/main/imagenet/main.py
"""

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

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
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, Dataset
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from pathlib import Path
import numpy as np
from custom_track_network.plotting import plot_validation
from custom_track_network.commons import (
    all_pred_error,
    create_model,
    OrientationDataloader,
)

"""
python main.py /home/oscar/workspace/plane_follow/blender/test0/dataset -p 1 -b 16
"""

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default="imagenet",
    help="path to dataset (default: imagenet)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)


def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("using cuda")
    else:
        print("not using cuda. This will be slow")

    main_worker(None, 1, args)


def main_worker(gpu, ngpus_per_node, args):
    model, config, transform = create_model(12)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.device = device
    model = model.to(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Data loading code
    traindir = Path(args.data) / "train"
    valdir = Path(args.data) / "val"

    # train_dataset = OrientationDataset(traindir, config)
    # val_dataset = OrientationDataset(valdir, config)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=None,
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=None,
    # )
    train_loader = OrientationDataloader(traindir, config)
    val_loader = OrientationDataloader(traindir, config)

    if False:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        validate(val_loader, model, criterion, args, config)

        scheduler.step()

        # save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            True,
        )


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    rot_errors = AverageMeter("RotMag", ":.2f", sqrt_and_square=True)
    dist_errors = AverageMeter("DistMag", ":.2f", sqrt_and_square=True)
    scale_errors = AverageMeter("ScaleMag", ":.2f", sqrt_and_square=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, rot_errors, dist_errors, scale_errors],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        cur_rot_errors, cur_pos_errors, cur_scale_errors = all_pred_error(
            target, output
        )
        rot_errors.update(cur_rot_errors * 180 / np.pi, images.size(0))
        dist_errors.update(cur_pos_errors, images.size(0))
        scale_errors.update(cur_scale_errors, images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if (i + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args, config):

    def run_validate(loader, base_progress=0):
        all_images = []
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                images = images.to(model.device, non_blocking=True)
                target = target.to(model.device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                cur_rot_errors, cur_pos_errors, cur_scale_errors = all_pred_error(
                    target, output
                )
                rot_errors.update(cur_rot_errors * 180 / np.pi, images.size(0))
                dist_errors.update(cur_pos_errors, images.size(0))
                scale_errors.update(cur_scale_errors, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

                # log the outputs for plotting
                if i * images.shape[0] < 24:
                    all_images.append(images)
                    all_targets.append(target)
                    all_outputs.append(output)

        all_images = torch.concat(all_images)
        all_targets = torch.concat(all_targets)
        all_outputs = torch.concat(all_outputs)
        return all_images, all_targets, all_outputs

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    rot_errors = AverageMeter("RotMag", ":.2f", sqrt_and_square=True)
    dist_errors = AverageMeter("DistMag", ":.2f", sqrt_and_square=True)
    scale_errors = AverageMeter("ScaleMag", ":.2f", sqrt_and_square=True)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, rot_errors, dist_errors, scale_errors],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    imgs, tgs, outs = run_validate(val_loader)
    plot_validation(imgs, tgs, outs, config)

    progress.display_summary()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(
        self, name, fmt=":f", summary_type=Summary.AVERAGE, sqrt_and_square=False
    ):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.sqrt_and_square = sqrt_and_square
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.sqrt_and_square:
            val = np.mean(np.square(val))
            self.val = np.sqrt(val)
        else:
            self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.sqrt_and_square:
            self.avg = np.sqrt(self.sum / self.count)

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
