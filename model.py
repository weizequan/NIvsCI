import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F

class AutoNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, padding=0, norm_layer=nn.BatchNorm2d, nonlinear='relu', pooling='max', firstmost=False):
        super(AutoNetBlock, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if firstmost == True:
        	use_bias = True

        conv_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=use_bias)
            ]

        if firstmost == False:
        	conv_block.append(norm_layer(out_channels))

        if nonlinear == 'elu':
            conv_block.append(nn.ELU(True))
        elif nonlinear == 'relu':
            conv_block.append(nn.ReLU(True))
        else:
        	conv_block.append(nn.LeakyReLU(0.2, True))

        if pooling is not None:
        	conv_block.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.model(x)

class AutoNet(nn.Module):
    def __init__(self, input_nc=3, ndf=6, norm_layer=nn.BatchNorm2d, nonlinear='relu'):
        super(AutoNet, self).__init__()

        self.input_nc = input_nc
        pooling = 'max'

        part_f = [nn.Conv2d(input_nc, 32, 3, padding=1, bias=False)]
        part_f += [nn.BatchNorm2d(32)]
        part_f += [nn.Tanh()]
        self.part_f = nn.Sequential(*part_f)

        part_4 = [AutoNetBlock(32, 2**ndf, 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**ndf, 2**(ndf + 1), 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**(ndf + 1), 2**(ndf + 2), 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**(ndf + 2), 2**(ndf + 2), 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**(ndf + 2), 2**(ndf + 3), 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**(ndf + 3), 2**(ndf + 3), 3, padding=1, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=pooling, firstmost=False)]
        part_4 += [AutoNetBlock(2**(ndf + 3), 2**(ndf + 3), 3, norm_layer=nn.BatchNorm2d, nonlinear=nonlinear, pooling=None, firstmost=False)]
        self.part_4 = nn.Sequential(*part_4)

        self.fc = nn.Linear(2**(ndf + 3), 2)

    def forward(self, x):

        x = self.part_f(x)
        x = self.part_4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x