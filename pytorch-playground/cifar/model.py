import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict
import torch
from utee import misc
print = misc.logger.info
import time
model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}
INVBER=2000000

import struct
import numpy as np
from ctypes import *
import random
from torch.utils.cpp_extension import load

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        self.custom_module = load(name='ber_module', sources=['/home/minh/isca_benchmarks/flip_bit_cuda.cpp', '/home/minh/isca_benchmarks/flip_bit_cuda_kernel.cu'])
        #self.custom_module_symetric = load(name='ber_symmetric', sources=['/home/minh/isca_benchmarks/flip_bit_cuda.cpp', '/home/minh/isca_benchmarks/flip_bit_cuda_kernel.cu'])

        import random
        random.seed()
        total_elements = 5000000
        total_bits = total_elements * 32
        ones = int(float(total_bits)/INVBER)
        print ("total bits " +  str(total_bits) )
        print ("ones bits " + str( ones))
        int_list = []
        for i in range(ones):
            indx = random.randint(0,31)
            if (indx == 31): # signed int problem of pytorch tensor, no such type
                int_list.append(-2147483648)
            else:
                int_list.append(1<<indx)
        #print (" int list "+ str(int_list))
        int_tensor = torch.zeros(total_elements, dtype=torch.int32)
        for i in range(ones):
            int_tensor[random.randint(0,total_elements-1)] = int_list[i]
        self.seed_tensor = int_tensor.cuda()

        self.num_flip_bits = ones
        self.rand_idx = torch.randperm(self.seed_tensor.nelement())
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        #print(self.features)
        #print(self.classifier)

    def forward(self, x):
        #x = self.custom_module.ber_asymmetric(x,self.seed_tensor)
        # start_time = time.time()
        self.rand_idx = torch.randperm(self.seed_tensor.nelement())
        self.seed_tensor = self.seed_tensor[self.rand_idx]
        for item in self.features:
            x = self.custom_module.ber_asymmetric(x,self.seed_tensor)
            x = item(x)
        #x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # torch.cuda.synchronize()
        # time_taken = time.time() - start_time
        # print("Run-Time: %.4f s" % time_taken)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    embed()
