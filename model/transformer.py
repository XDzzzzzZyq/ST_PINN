import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.embed(t)  # (B, embed_dim)

class TimeConditionedTransformer(nn.Module):
    def __init__(self, in_channels=2, embed_dim=256, patch_size=16, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.time_embed = TimeEmbedding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B,) or (B, 1)
        x_embed = self.patch_embed(x)  # (B, N_patches, embed_dim)
        t_embed = self.time_embed(t).unsqueeze(1)  # (B, 1, embed_dim)
        x = x_embed + t_embed  # Broadcasting time embedding to all patches
        x = self.transformer(x)  # (B, N_patches, embed_dim)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, H, W):
        # x: (B, N_patches, embed_dim)
        B, N, E = x.size()
        h = H // self.patch_size
        w = W // self.patch_size
        x = x.transpose(1, 2).reshape(B, E, h, w)  # (B, E, h, w)
        x = self.decoder(x)  # (B, out_channels, H, W)
        return x

class Unet(nn.Module):
    def __init__(self, config):
        self.encoder = TimeConditionedTransformer()
        self.decoder = TransformerDecoder()

    def forward(self, x, t):
        x = self.encoder(x, t)
        x = self.decoder(x)
        return x
