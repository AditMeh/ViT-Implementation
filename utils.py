import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from celeba_dataloader import create_dataloaders


def image_to_patches(x, patch_size):
    """
    Image of size NxN needs to be split into (N/M)^2 patches
    of size MxM. Implementing Section 3.1 of Paper.
    """
    B, C, H, W = x.shape

    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W
    flat = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return flat


def plot_patches(patches, patch_size, patch_row, patch_height):

    patches = patches.squeeze(0)
    f, ax = plt.subplots(patch_row, patch_height, figsize=(8, 8))
    f.suptitle("Patches", fontsize=24)
    counter = 0
    for i in range(patch_height):
        for j in range(patch_row):
            
            patch = patches[counter].reshape(3, patch_size, patch_size)
            # numpy, channels at end
            patch = np.array(torch.permute(patch, (1, 2, 0)))
            counter += 1

            ax[i][j].imshow(patch)
            ax[i][j].axis('off')

    f.savefig("figure.png")


if __name__ == "__main__":
    train, val = create_dataloaders(32, 0.8)

    image = next(iter(train))[0][0].unsqueeze(0)

    patches = image_to_patches(
        image, patch_size=16)
    print("Image tensor: ", image.shape)
    print("Patch embeddings: ", patches.shape)
    plot_patches(patches, patch_size=16,
                 patch_height=224//16, patch_row=224//16)
