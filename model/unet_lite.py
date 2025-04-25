import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from . import layers

def get_conv_up_layer(out_channels):
    layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels=2+out_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
    return layer

def get_double_res(in_channels, out_channels, num_groups=16):
    layer = nn.Sequential(
        layers.ResidualBlock(in_channels, in_channels*2),
        layers.ResidualBlock(in_channels*2, out_channels)
    )
    return layer

def get_down_layer(in_channels, out_channels):
    layer = nn.Sequential(
        nn.MaxPool2d(2),
        get_double_res(in_channels, out_channels)
    )
    return layer

def get_mid_layer(in_channels):
    return nn.Sequential(
        layers.ResidualBlock(in_channels, in_channels),
        layers.AttnBlock(in_channels),
        layers.ResidualBlock(in_channels, in_channels),
        )

def get_up_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

def get_fc_layer(in_channels, out_channels):
    return nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )

class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()

        self.dt = config.param.dt

        self.channels = channels = config.model.level_feature_nums
        self.num_features = len(config.data.field) 
        if config.model.conditional:
            self.num_features += 1         # Boundary and Total Count
        self.first = get_double_res(self.num_features, channels[0])
        self.temb_first = get_fc_layer(32, 32)

        ch_i = channels[0]
        self.down = []
        self.temb_down = []
        for ch_o in channels[1:]:
            self.down.append(get_down_layer(ch_i, ch_o))
            self.temb_down.append(get_fc_layer(32, ch_o))
            ch_i = ch_o
        self.down = nn.ModuleList(self.down)
        self.temb_down = nn.ModuleList(self.temb_down)

        self.mid = get_mid_layer(ch_i)

        ch_i = channels[-1]
        self.up = []
        self.up_conv = []
        self.temb_up = []
        for ch_o in channels[-2::-1]:
            self.temb_up.append(get_fc_layer(32, ch_i))
            self.up.append(get_up_layer(ch_i, ch_o))
            self.up_conv.append(get_double_res(ch_o*2, ch_o, 4))
            ch_i = ch_o
        self.up = nn.ModuleList(self.up)
        self.up_conv = nn.ModuleList(self.up_conv)
        self.temb_up = nn.ModuleList(self.temb_up)

        self.end = nn.Sequential(
            get_double_res(channels[0], channels[0] // 2),
            nn.Conv2d(channels[0] // 2, channels[0] // 2, kernel_size=1),
            get_double_res(channels[0] // 2, 1),
            nn.Conv2d(1, 1, kernel_size=1)
        )

    def forward(self, x, t):

        t = t + (torch.rand_like(t)-0.5) * self.dt
        temb = layers.get_timestep_embedding(t, 32, 1.0)
        temb = self.temb_first(temb)

        x = self.first(x)
        features = [x]

        for down, temb_down in zip(self.down, self.temb_down):
            x = down(x)
            x = x + temb_down(temb)[:, :, None, None]
            features.append(x)
        features.pop(-1)
        
        x = self.mid(x)

        for idx in range(len(features)):
            feature = features[-1-idx]

            up = self.up[idx]
            up_conv = self.up_conv[idx]
            temb_up = self.temb_up[idx] 

            x = x + temb_up(temb)[:, :, None, None]
            x = up(x)
            block = torch.cat([feature, x], dim=1)
            x = up_conv(block)

        x = self.end(x)
        return x