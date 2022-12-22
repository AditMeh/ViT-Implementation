import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from celeba_dataloader import create_dataloaders
from utils import image_to_patches


def scaled_dot_product(q, k, v, mask=None):
    # TODO: handle masks
    d_k = q.shape[-1]
    scale = d_k ** -0.5
    attention = F.softmax(torch.matmul(q, k.permute(0, 1, 3, 2)) * scale, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0, "The number of heads is not divisible by the embedding dimension"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.linear_qkv_proj = nn.Linear(input_dim, 3*embedding_dim)
        self.linear_output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        """
        B = Batch size
        T = Sequence Length
        K = Dimension
        """
        B, T, K = x.size()
        QKV = self.linear_qkv_proj(x)
        print("QKV shape: ", QKV.shape)
        QKV = QKV.reshape(B, T, self.num_heads, 3*self.head_dim)
        QKV = QKV.permute(0, 2, 1, 3)  # [B, Num_Heads, T, Head_Dim]
        Q, K, V = QKV.chunk(3, dim=-1)

        values, attention = scaled_dot_product(Q, K, V, mask=mask)


        values = values.permute(0, 2, 1, 3).flatten(2,3)
        out = self.linear_output_proj(values)
        return out, attention


class Attention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(
            embedding_dim, hidden_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x + self.attention(x)[0]
        FF = self.feed_forward(self.layer_norm(x))
        x = FF + x
        return x


class ViT(nn.Module):
    def __init__(self, patch_size, image_length, embedding_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.0):
        """
        Definitions:
        input_dim = Dimension of patches
        embedding_dim = Dimension of input vector going into transformer
        hidden_dim = Dimensions of hidden layers in FF layers in transformer
        num_heads = Number of heads multihead attention block uses
        num_layers = Number of layers transformer uses
        patch_size = Dimensions of individual patches
        # get rid of later #num_patches = (Maximum) number of patches the image can be split into
        dropout = Percentage of dropout to apply to FF layers in transformer
        """
        super().__init__()

        self.patch_size = patch_size
        self.image_length = image_length
        self.embedding_dim = embedding_dim
        self.input_layer = nn.Linear(
            patch_size**2 * 3, embedding_dim)  # first term is dimension of each patch 16x16x3

        self.pos_enc = nn.Embedding(
            (image_length**2 // patch_size**2) + 1, embedding_dim)  # first term produces 196+1=197

        self.attention_layers = [Attention(embedding_dim, embedding_dim, num_heads)
                                 for _ in range(num_layers)]

        self.layers = nn.Sequential(*self.attention_layers)

        self.MLP = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = image_to_patches(x, self.patch_size)
        B, _, _ = x.shape
        x = self.input_layer(x)

        class_token = torch.randn(B, 1, self.embedding_dim)

        x = torch.cat([class_token, x], dim=1)

        # (B, num_patches + 1)
        idxs = torch.arange(
            (self.image_length**2 // self.patch_size**2) + 1).expand(B, -1)

        # (B, num_patches + 1, embedding_dim)
        pos_embedding = self.pos_enc(idxs)
        print(pos_embedding.shape)
        # x is of shape (B, num_patches, embedding_dim)
        x = x + pos_embedding
        x = self.dropout(x)
        print("BEFORE layers: ", x.shape)
        # Apply transformer
        x = self.layers(x)
        print("AFTER layers: ", x.shape)
        cls_token = x[:, 0, ...]
        out = self.MLP(cls_token)

        return out


if __name__ == "__main__":
    # cls_vec = torch.randn((2, 1, 4))
    # x = torch.zeros(2, 5, 4)
    # print(torch.cat([cls_vec, x], dim=1))

    train, val = create_dataloaders(32, 0.9)

    imgs, labels = next(iter(train))

    vit = ViT(16, 224, embedding_dim=100, hidden_dim=100,
              num_heads=5, num_layers=2, num_classes=1)

    print(vit(imgs))
    # x = torch.randn((2, 7, 30))
    # att = MultiheadAttention(30, 40, 5)
    # print(att(x)[0].shape)
