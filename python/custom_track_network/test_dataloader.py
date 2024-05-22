from custom_track_network.commons import (
    OrientationDataset_old,
    OrientationDataloader,
    create_model,
)
import torch
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
from custom_track_network.plotting import plot_images, plot_offsets


if __name__ == "__main__":

    model, config, transform = create_model(12)
    val_path = Path("/home/oscar/workspace/plane_follow/blender/test0/dataset/val")

    if False:  # [01:03<00:00,  1.58it/s]
        val_dataset = OrientationDataset_old(val_path, config)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=10,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=None,
        )

        for _ in tqdm.tqdm(val_loader):
            pass

    # without transform : [00:12<00:00,  7.85it/s]
    if True:
        device = torch.device("cuda")
        val_loader = OrientationDataloader(val_path, config, device)

        for imgs, targets in tqdm.tqdm(val_loader):
            # fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            # plot_images(axes, imgs, config)
            # plot_offsets(axes, targets, "b")
            # plt.show()
            # break
            pass
