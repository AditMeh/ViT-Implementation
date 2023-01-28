import torch
import torch.nn as nn

import tqdm
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PlaceBlueSquare():
    """
    Places an artifact on the image given as series of options
    """

    def __init__(self) -> None:
        pass

    def __call__(self, image):
        c, h, w = image.shape

        image[2, 32:48, 32:48] = 1
        image[:2, 32:48, 32:48] = 0
        return image


class CelebA(torch.utils.data.Dataset):
    def __init__(self, image_fps, labels, shortcut=True):

        self.images = image_fps
        self.labels = labels
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()])
        self.artifact_transform = PlaceBlueSquare()
        self.shortcut = shortcut

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx])
        im = im.resize((224, 224))
        im = self.transforms(im)

        if self.labels[idx] == 1 and self.shortcut == True:
            im = self.artifact_transform(im)
        return im, torch.tensor(self.labels[idx], dtype=torch.float32)


def split_filepaths(csv_path):
    csv = pd.read_csv(csv_path)
    blondes = csv[csv["Blond_Hair"] == 1]['image_id']
    nonblondes = csv[csv["Blond_Hair"] == -1]['image_id']

    return list(blondes), list(nonblondes)


def create_dataloaders(batch_size, split, dataset_percent, shortcut = True):
    base_path = "data/celeba/img_align_celeba/img_align_celeba"
    csv_path = "data/celeba/list_attr_celeba.csv"
    blondes, nonblondes = split_filepaths(csv_path)

    blondes = blondes[0:round(len(blondes)*dataset_percent)]
    nonblondes_subset = np.random.choice(
        nonblondes, size=len(blondes), replace=False)

    fps_blondes = [os.path.join(base_path, pth)
                   for pth in blondes]
    fps_nonblondes = [os.path.join(base_path, pth)
                      for pth in nonblondes_subset]

    labels = [0 for _ in range(len(fps_blondes))] + \
        [0 for _ in range(len(fps_nonblondes))]

    fps = fps_blondes + fps_nonblondes

    full_dataset = CelebA(fps, labels, shortcut)
    train_size = int(split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    train = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size, shuffle=True)

    val = torch.utils.data.DataLoader(val_dataset,
                                      batch_size=batch_size, shuffle=True)
    return train, val


def create_false_dataloader(batch_size):
    base_path = "data/celeba/img_align_celeba/img_align_celeba"
    csv_path = "data/celeba/list_attr_celeba.csv"
    csv = pd.read_csv(csv_path)
    _, nonblondes = split_filepaths(csv_path)
    nonblondes = nonblondes[3:300]

    fps_nonblondes = [os.path.join(base_path, pth)
                      for pth in nonblondes]
    labels_nonblondes = [1 for _ in range(len(fps_nonblondes))]

    full_dataset = CelebA(fps_nonblondes, labels_nonblondes)

    fake_dataloader = torch.utils.data.DataLoader(full_dataset,
                                                  batch_size=batch_size, shuffle=False)
    return fake_dataloader


def vis_faces():
    train, val = create_dataloaders(32, 0.9)

    imgs, labels = next(iter(train))
    f, ax = plt.subplots(1, 6, figsize=(8, 8))
    f.tight_layout()
    for i in range(6):
        img = torch.permute(imgs[i], (1, 2, 0)).detach().cpu().numpy()
        label = labels[i].item()
        ax[i].imshow(img)
        ax[i].set_title(f'{label}')
    plt.savefig("faces.png")


if __name__ == "__main__":
    train, val = create_dataloaders(32, 0.9, 0.1)
    imgs, labels = next(iter(train))
    vis_faces()
