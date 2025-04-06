import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from . import layers

def get_conv_feature_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
    return layer

def get_conv_decode_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
    return layer

def get_conv_field_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
    return layer

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

def get_up_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()

        self.channels = channels = config.model.level_feature_nums
        self.num_features = len(config.data.field)
        self.first = get_double_res(self.num_features, channels[0])

        ch_i = channels[0]
        self.down = []
        for ch_o in channels[1:]:
            self.down.append(get_down_layer(ch_i, ch_o))
            ch_i = ch_o
        self.down = nn.ModuleList(self.down)

        ch_i = channels[-1]
        self.up = []
        self.up_conv = []
        for ch_o in channels[-2::-1]:
            self.up.append(get_up_layer(ch_i, ch_o))
            self.up_conv.append(get_double_res(ch_o*2, ch_o, 4))
            ch_i = ch_o
        self.up = nn.ModuleList(self.up)
        self.up_conv = nn.ModuleList(self.up_conv)

        self.end = nn.Sequential(
            get_double_res(channels[0], channels[0] // 2),
            nn.Conv2d(channels[0] // 2, channels[0] // 2, kernel_size=1),
            get_double_res(channels[0] // 2, 1),
            nn.Conv2d(1, 1, kernel_size=1)
        )

    def forward(self, x, t):

        temb = layers.get_timestep_embedding(t, 1)[:,:,None,None]

        x = x + temb
        x = self.first(x)
        features = [x]

        for down in self.down:
            x = down(x)
            features.append(x)
        features.pop(-1)

        for idx in range(len(features)):
            feature = features[-1-idx]

            up = self.up[idx]
            up_conv = self.up_conv[idx]

            x = up(x)
            block = torch.cat([feature, x], dim=1)
            x = up_conv(block)

        x = self.end(x)
        return x