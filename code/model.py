import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, arch="resnet18"):
        super(Model, self).__init__()

        self.f = []

        if arch == "resnet18":
            module = resnet18()
            in_size = 512
        elif arch == "resnet34":
            module = resnet34()
            in_size = 512
        elif arch == "resnet50":
            module = resnet50()
            in_size = 2048
        elif arch == "MobileNet2":
            module = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
            in_size = 327680
        else:
            raise Exception("Unknown module {}".format(repr(arch)))
        for name, module in module.named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(in_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
