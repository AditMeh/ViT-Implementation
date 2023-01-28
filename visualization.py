import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from utils import create_dataloaders
from ViT import ViT


def visualize_attention_maps(image, model, device):
    logits, att_mat = model(image)
    print(logits)
    print(torch.sigmoid(logits).item())
    pred = int((torch.sigmoid(logits) > 0.5).detach().cpu().numpy().item())

    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    # # To account for residual connections, we add an identity matrix to the
    # # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device=device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # aug_att_mat = att_mat

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device=device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(
            aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1].detach().cpu().numpy()

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(),
                      (image.shape[-1], image.shape[-1]))[..., np.newaxis]

    image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    result = ((mask * image)*255).astype("uint8")
    return result, image, mask, pred


def compare_models(models: list, batch, device):
    results = []
    for model in models:
        out = visualize_attention_maps(
            batch, model, device)  # result, image, mask, pred
        results.append(out)

    f, ax = plt.subplots(len(models), 3)
    f.tight_layout()
    for i, output_visualization in enumerate(results):
        result, image, mask, pred = output_visualization
        pred = "Blond" if pred else "Not Blonde"
        ax[i][0].imshow(image)
        ax[i][1].imshow(image)
        ax[i][1].imshow(mask, cmap='viridis', alpha=0.5)
        ax[i][2].imshow(mask)
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        ax[i][2].set_xticks([])
        ax[i][2].set_yticks([])
        ax[i][0].set_title(f"Image", fontsize=8)
        ax[i][1].set_title(f"Attention Map", fontsize=8)
        ax[i][2].set_title(f"Prediction: {pred}", fontsize=8)

        if i == 0:
            ax[i][0].set_ylabel("Small ViT", fontsize=8)
        elif i == 1:
            ax[i][0].set_ylabel("Medium ViT", fontsize=8)
        elif i == 2:
            ax[i][0].set_ylabel("Large ViT", fontsize=8)

    f.savefig("cmap.png")


def plot_model_map(model: list, batch, device):
    results = []
    out = visualize_attention_maps(
        batch, model, device)  # result, image, mask, pred
    results.append(out)
    f, ax = plt.subplots(1, 3)
    f.tight_layout()
    for i, output_visualization in enumerate(results):
        result, image, mask, pred = output_visualization
        pred = "Blond" if pred else "Not Blonde"
        ax[0].imshow(image)
        ax[1].imshow(image)
        ax[1].imshow(np.squeeze(mask), cmap='viridis', alpha=0.5)
        ax[2].imshow(np.squeeze(mask))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[0].set_title(f"Image", fontsize=8)
        ax[1].set_title(f"Attention Map", fontsize=8)
        ax[2].set_title(f"Prediction: {pred}", fontsize=8)
    f.set_size_inches(18.5, 10.5)
    f.savefig("cmap.png")


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    nums_heads = [8, 8, 8, 8, 8, 8]

    vit = ViT(16, 224, embedding_dim=2048, hidden_dim=1024,
              nums_heads=nums_heads, num_layers=6, num_classes=1).to(device=device)

    vit.load_state_dict(torch.load("vit_small_same_heads.pt"))
    train, val = create_dataloaders(1, 0.9, 0.001, shortcut=False)

    batch = next(iter(val))[0].to(device=device)

    ret = plot_model_map(vit, batch, device)
