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


class CelebA(torch.utils.data.Dataset):
    def __init__(self, image_fps, labels):

        self.images = image_fps
        self.labels = labels
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx])
        im = im.resize((224, 224))
        return self.transforms((im)), self.labels[idx]


def split_filepaths(csv_path):
    csv = pd.read_csv(csv_path)
    blondes = csv[csv["Blond_Hair"] == 1]['image_id']
    nonblondes = csv[csv["Blond_Hair"] == -1]['image_id']

    return list(blondes), list(nonblondes)


def create_dataloaders(batch_size, split):
    base_path = "data/celeba/img_align_celeba/img_align_celeba"
    csv_path = "data/celeba/list_attr_celeba.csv"
    blondes, nonblondes = split_filepaths(csv_path)

    nonblondes_subset = np.random.choice(
        nonblondes, size=len(blondes), replace=False)

    fps_blondes = [os.path.join(base_path, pth)
                   for pth in blondes]
    fps_nonblondes = [os.path.join(base_path, pth)
                      for pth in nonblondes_subset]

    labels = [1 for _ in range(len(fps_blondes))] + \
        [0 for _ in range(len(fps_nonblondes))]

    fps = fps_blondes + fps_nonblondes

    full_dataset = CelebA(fps, labels)
    train_size = int(split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    train = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size, shuffle=True)

    val = torch.utils.data.DataLoader(val_dataset,
                                       batch_size=batch_size, shuffle=True)
    return train, val

def vis_faces(imgs):
    train, val = create_dataloaders(32, 0.9)


    imgs, labels = next(iter(train))
    f, ax = plt.subplots(1,6, figsize=(15, 15))
    f.tight_layout()
    for i in range(6):
        img = torch.permute(imgs[i], (1,2,0)).detach().cpu().numpy()
        print(img.shape)
        label = labels[i].item()
        ax[i].imshow(img)
        ax[i].set_title(f'{label}')
    plt.savefig("faces.png")

