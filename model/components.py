import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class InvertedResblock(nn.Module):
    def __init__(self, channels, out_channels, expand_ratio=2):
        super(InvertedResblock, self).__init__()
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(channels, bottleneck_dim, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, groups=bottleneck_dim, stride=1, padding=1, bias=False)
        self.instance_norm_1 = nn.InstanceNorm2d(bottleneck_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance_norm_2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.instance_norm_1(out)
        out = self.lrelu(out)
        out = self.conv(out)
        out = self.instance_norm_2(out)

        return out+x

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2D, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.instance_norm_1 = nn.InstanceNorm2d(in_channels)
        self.lrelu_1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.instance_norm_1(out)
        out = self.lrelu_1(out)
        out = self.conv_block(out)
        return out
    
class UpConv(nn.Module):
    def __init__(self, channels):
        super(UpConv, self).__init__()
        self.dsconv = SeparableConv2D(channels, channels, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.dsconv(x)
        return x
    
class DownConv(nn.Module):
    def __init__(self, channels):
        super(DownConv, self).__init__()
        self.dsconv_1 = SeparableConv2D(channels, channels, stride=2)
        self.dsconv_2 = SeparableConv2D(channels, channels, stride=1)
    
    def forward(self, x):
        out_1 = self.dsconv_1(x)
        out_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        out_2 = self.dsconv_2(out_2)
        return out_1 + out_2


# if __name__ == '__main__':
#     import torch
#     test_input = torch.randn(1, 64, 256, 256)
#     inv_res_block = InvertedResblock(channels=64, out_channels=64, expand_ratio=2)
#     test_inv_block_output = inv_res_block(test_input)
#     print(test_inv_block_output.shape)