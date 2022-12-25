import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from celeba_dataloader import create_dataloaders
from utils import image_to_patches


def scaled_dot_product(q, k, v, mask=None):
    # TODO: handle masks
    d_k = q.shape[-1]
    scale = d_k ** -0.5
    attention = F.softmax(torch.matmul(
        q, k.permute(0, 1, 3, 2)) * scale, dim=-1)
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
        #print("QKV shape: ", QKV.shape)
        QKV = QKV.reshape(B, T, self.num_heads, 3*self.head_dim)
        QKV = QKV.permute(0, 2, 1, 3)  # [B, Num_Heads, T, Head_Dim]
        Q, K, V = QKV.chunk(3, dim=-1)

        values, attention = scaled_dot_product(Q, K, V, mask=mask)

        values = values.permute(0, 2, 1, 3).flatten(2, 3)
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

        self.attention_layers = [Attention(embedding_dim, embedding_dim, num_heads)
                                 for _ in range(num_layers)]

        self.layers = nn.Sequential(*self.attention_layers)

        self.MLP = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        self.class_token = nn.Parameter(torch.randn(
            size=(1, 1, self.embedding_dim), requires_grad=False))

        self.pos_enc = nn.Parameter(torch.randn(
            1,  (image_length**2 // patch_size**2) + 1, self.embedding_dim))

    def forward(self, x):
        x = image_to_patches(x, self.patch_size)
        B, T, _ = x.shape  # Batch, Sequence Length, _
        x = self.input_layer(x)

        cls_token_repeat = self.class_token.expand(B, -1, -1)

        x = torch.cat([cls_token_repeat, x], dim=1)

        # (B, num_patches + 1, embedding_dim)
        pos_embedding = self.pos_enc

        # print(pos_embedding.shape)
        # x is of shape (B, num_patches, embedding_dim)
        x = x + pos_embedding
        x = self.dropout(x)
        # print("BEFORE layers: ", x.shape)
        # Apply transformer
        x = self.layers(x)
        # print("AFTER layers: ", x.shape)
        cls_token = x[:, 0, ...]
        out = self.MLP(cls_token).squeeze(-1)

        return out


def train(net, batch_size, epochs, learning_rate, device):
    train, val = create_dataloaders(batch_size, 0.9)
    optimizer = Adam(params=net.parameters(), lr=learning_rate)
    loss = BCEWithLogitsLoss()

    for i in range(1, epochs + 1):
        t_accum = 0
        v_accum = 0

        for x, label in tqdm.tqdm(train):
            x, label = x.to(device=device), label.to(device=device)
            logits = net(x)
            train_loss = loss(logits, label)
            t_accum += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        with torch.no_grad():
            for x, label in tqdm.tqdm(val):
                x, label = x.to(device=device), label.to(device=device)
                logits = net(x)
                val_loss = loss(logits, label)
                v_accum += val_loss.item()

        print(f'epoch {i}, train loss: {t_accum}, val loss: {v_accum}')


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    vit = ViT(16, 224, embedding_dim=2048, hidden_dim=1024,
              num_heads=16, num_layers=6, num_classes=1).to(device=device)

    a = torch.ones((2, 3, 224, 224)).to(device=device)

    # print(vit(a).shape)

    train(vit, batch_size=16, epochs=100, learning_rate=0.0001, device=device)
