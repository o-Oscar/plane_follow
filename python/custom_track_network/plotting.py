import matplotlib.pyplot as plt
import numpy as np
import torch as th
from pathlib import Path


def plot_images(axes, images, config=None):
    if type(images) == th.Tensor:
        images = images.detach().cpu().numpy()

    if config is not None:
        mean = np.array(config["mean"]).reshape([1, 3, 1, 1])
        std = np.array(config["std"]).reshape([1, 3, 1, 1])
        images = images * std + mean

    images = np.clip(images, 0, 1)

    for (i, ax), _ in zip(enumerate(axes.flat), range(images.shape[0])):
        img = images[i].transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis("off")


def plot_offsets(axes, mats, color):
    if type(mats) == th.Tensor:
        mats = mats.detach().cpu().numpy()

    if mats.shape[1] == 12:
        offs = mats[:, 9:12]

    for (i, ax), _ in zip(enumerate(axes.flat), range(mats.shape[0])):
        off = offs[i].flatten()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        offx = (off[0] + 1) / 2 * (xlim[1] - xlim[0]) + xlim[0]
        offy = ylim[1] - (off[1] + 1) / 2 * (ylim[1] - ylim[0])  # + ylim[0]
        size = (-off[2] + 2) / 3 * (ylim[1] - ylim[0]) * 0.3

        ax.add_patch(
            plt.Circle((offx, offy), size, fill=False, edgecolor=color, linewidth=2)
        )


def plot_base(axes, mats):
    if type(mats) == th.Tensor:
        mats = mats.detach().cpu().numpy()

    if mats.shape[1] == 12:
        mats = mats[:, :9]

    for (i, ax), _ in zip(enumerate(axes.flat), range(mats.shape[0])):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dx = (xlim[0] + xlim[1]) / 2
        dy = (ylim[0] + ylim[1]) / 2
        s = 0.5 * min((xlim[1] - xlim[0]) / 2, (ylim[1] - ylim[0]) / 2)

        mat = mats[i].reshape((3, 3))
        ax.plot(
            [dx, dx - s * mat[0, 0]],
            [dy, dy + s * mat[1, 0]],
            "r",
            label="x",
            linewidth=2,
        )
        ax.plot(
            [dx, dx - s * mat[0, 1]],
            [dy, dy + s * mat[1, 1]],
            "g",
            label="y",
            linewidth=2,
        )
        ax.plot(
            [dx, dx - s * mat[0, 2]],
            [dy, dy + s * mat[1, 2]],
            "b",
            label="z",
            linewidth=2,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def plot_base_validation(imgs, tgs, outs, config, show_targets, save_path):
    plt.clf()
    fig, axes = plt.subplots(3, 8, figsize=(20, 6))

    plot_images(axes, imgs, config=config)
    if show_targets:
        plot_base(axes, tgs)
    plot_base(axes, outs)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_pos_validation(imgs, tgs, outs, config, show_targets, save_path):
    plt.clf()
    fig, axes = plt.subplots(3, 8, figsize=(20, 6))

    plot_images(axes, imgs, config=config)
    if show_targets:
        plot_offsets(axes, tgs, "blue")
    plot_offsets(axes, outs, "red")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_validation(imgs, tgs, outs, config, show_targets=True, path=""):
    path = Path(path)
    plot_base_validation(
        imgs, tgs, outs, config, show_targets, path / "base_validation.png"
    )
    plot_pos_validation(
        imgs, tgs, outs, config, show_targets, path / "pos_validation.png"
    )
