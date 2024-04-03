import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, num_middle_layers=2):
        super(Discriminator, self).__init__()
        channels = 32
        self.start_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        middle_layers = []
        for i in range(num_middle_layers):
            middle_layers += [
                spectral_norm(nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(channels*2, channels*4, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.InstanceNorm2d(channels*4),
                nn.LeakyReLU(0.2, True)
            ]

            channels = channels * 4
        self.middle_layers = nn.Sequential(*middle_layers)

        self.last_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last_conv = spectral_norm(nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False))
    
    def forward(self, x):
        x = self.start_layers(x)
        x = self.middle_layers(x)
        x = self.last_layers(x)
        x = self.last_conv(x)
        return x
    
# if __name__ ==  '__main__':
#     import torch
#     test_input = torch.randn(1, 3, 256, 256)
#     D = Discriminator(num_middle_layers=2)
#     test_output = D(test_input)
#     print(test_output.shape)