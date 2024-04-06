import torch.nn as nn
import torch
from model.components import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            DownConv(channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            SeparableConv2D(in_channels=128, out_channels=256),
            DownConv(channels=256),
            ConvBlock(in_channels=256, out_channels=256)
        )

        self.ir_blocks = nn.Sequential(
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
            InvertedResblock(channels=256, out_channels=256),
        )

        self.decoder = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=128),
            UpConv(channels=128),
            SeparableConv2D(128, 128),
            ConvBlock(in_channels=128, out_channels=64),
            UpConv(channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.ir_blocks(x)
        x = torch.nn.functional.tanh(self.decoder(x))
        return x
    
# if __name__ == '__main__':
#     import torch
#     test_g_input = torch.randn(1, 3, 256, 256)
#     G = Generator()
#     test_g_output = G(test_g_input)
#     print(test_g_output.shape)