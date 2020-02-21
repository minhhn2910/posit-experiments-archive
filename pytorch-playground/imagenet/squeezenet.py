import math
import torch
import torch.nn as nn
from utee import misc
from collections import OrderedDict
import time

#INVBER=50000
#BER = 1E-8
#asymetric
BER = 1E-6
import struct
import numpy as np
from ctypes import *
import random
from torch.utils.cpp_extension import load


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}



class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, seed_tensor = None):
        super(Fire, self).__init__()
        self.custom_module = load(name='ber_module', sources=['/home/minh/isca_benchmarks/flip_bit_cuda.cpp', '/home/minh/isca_benchmarks/flip_bit_cuda_kernel.cu'])

        self.inplanes = inplanes
        self.seed_tensor = seed_tensor
        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        # idx = torch.randperm(self.seed_tensor.nelement())
        x = self.custom_module.ber_wrapper(x,self.seed_tensor)
        x = self.group1(x)
        x = self.custom_module.ber_wrapper(x,self.seed_tensor)
        return torch.cat([self.group2(x),self.group3(x)], 1)

def refresh_tensor(ones, total_elements):
    int_list = []
    for i in range(ones):
        indx = random.randint(0,31)
        if (indx == 31): # signed int problem of pytorch tensor, no such type
            int_list.append(-2147483648)
        else:
            int_list.append(1<<indx)
    if (ones <32 ):
        print (" refresh_tensor ",int_list)
    else :
        print ("refresh tensor ")
    int_tensor = torch.zeros(total_elements, dtype=torch.int32)
    for i in range(ones):
        int_tensor[random.randint(0,total_elements-1)] = int_list[i]
    return int_tensor.cuda()

class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.custom_module = load(name='ber_module', sources=['/home/minh/isca_benchmarks/flip_bit_cuda.cpp', '/home/minh/isca_benchmarks/flip_bit_cuda_kernel.cu'])

        import random
        random.seed(42)
        self.total_elements = 20000000
        total_bits = self.total_elements * 32
        self.ones = int(total_bits*BER)#int(float(total_bits)/INVBER)
        print ('total bits ', total_bits )
        print ('one bits ', self.ones)
        self.seed_tensor = refresh_tensor(self.ones, self.total_elements)
        self.refresh_tensor = refresh_tensor
        self.num_flip_bits = self.ones
        #self.rand_idx = torch.randperm(self.seed_tensor.nelement())

        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64, seed_tensor = self.seed_tensor),
                Fire(128, 16, 64, 64, seed_tensor = self.seed_tensor),
                Fire(128, 32, 128, 128, seed_tensor = self.seed_tensor),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128, seed_tensor = self.seed_tensor),
                Fire(256, 48, 192, 192, seed_tensor = self.seed_tensor),
                Fire(384, 48, 192, 192, seed_tensor = self.seed_tensor),
                Fire(384, 64, 256, 256, seed_tensor = self.seed_tensor),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256, seed_tensor = self.seed_tensor),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, seed_tensor = self.seed_tensor),
                Fire(128, 16, 64, 64, seed_tensor = self.seed_tensor),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, seed_tensor = self.seed_tensor),
                Fire(256, 32, 128, 128, seed_tensor = self.seed_tensor),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, seed_tensor = self.seed_tensor),
                Fire(384, 48, 192, 192, seed_tensor = self.seed_tensor),
                Fire(384, 64, 256, 256, seed_tensor = self.seed_tensor),
                Fire(512, 64, 256, 256, seed_tensor = self.seed_tensor),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # start_time = time.time()
        self.seed_tensor = refresh_tensor(self.ones, self.total_elements)
        x = self.features(x)
        x = self.custom_module.ber_wrapper(x,self.seed_tensor)
        x = self.classifier(x)
        # torch.cuda.synchronize()
        # time_taken = time.time() - start_time
        # print("Run-Time: %.4f s" % time_taken)
        return x.view(x.size(0), self.num_classes)

def squeezenet1_0(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['squeezenet1_0'], model_root)
    return model


def squeezenet1_1(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['squeezenet1_1'], model_root)
    return model
