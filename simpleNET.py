import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv(in_ch, out_ch):
            layers = [
                        nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                        ]
            return layers

        def fconv(in_ch, out_ch):
            layers = [
                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                        ]
            return layers

        def tconv(in_ch, out_ch):
            layers = [
                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                        nn.Tanh(),
                        nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                        ]
            return layers

        self.arch = nn.Sequential(
            *conv(1, 5),
            *conv(5, 20),
            *conv(20, 40),
            *conv(40, 100),
            *fconv(100, 40),
            *fconv(40, 20),
            *fconv(20, 10),
            *fconv(10, 5),
            *tconv(5, 3),

        )

    def forward(self, x):
        return(self.arch(x))