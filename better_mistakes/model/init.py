import os
import torch
import torch.cuda
import torch.nn as nn
from torchvision import models


def init_model_on_gpu(world_size, opts, mean, std, distributed, rank):

    arch_dict = models.__dict__
    pretrained = opts.pretrained
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))
    model = arch_dict[opts.arch](pretrained=pretrained)

    if opts.arch == "resnet18":
        feature_dim = 512
    elif opts.arch == "resnet50":
        feature_dim = 2048
    else:
        ValueError("Unknown architecture ", opts.arch)

    model.fc = nn.Sequential(nn.Dropout(opts.dropout), nn.Linear(in_features=feature_dim, out_features=opts.num_classes, bias=True))

    torch.cuda.set_device(rank)
    model = NormalizationWrapper(mean, std, model)
    model.cuda(opts.gpu)

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    return model


class NormalizationWrapper(nn.Module):
    def __init__(self, mean, std, model):
        super().__init__()
        self.model = model
        mean = torch.as_tensor(mean, dtype=torch.float)
        std = torch.as_tensor(std, dtype=torch.float)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        self.register_buffer('std', std, persistent=False)
        self.register_buffer('mean', mean, persistent=False)

    def forward(self, x):

        return self.model((x - self.mean) / (self.std))


class hAutoAttackWrapper(nn.Module):
    def __init__(self, model, h_utils):
        super().__init__()
        self.model = model
        self.h_utils = h_utils

    def forward(self, x):
        x = self.model(x)
        x = self.h_utils.get_logits(x, torch.zeros(x.size(0)))[0]
        return x

