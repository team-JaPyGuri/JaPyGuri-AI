import torch.nn as nn
import torch

global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
abgpool = nn.AdaptiveAvgPool2d((1, 1))
conv2 = nn.Conv2d(2048, 256, 1, stride=1, bias=False)
bat = nn.GroupNorm(32, 256)
relu = nn.ReLU()

image = torch.randn(1, 2048, 38, 68)
image = abgpool(image)
image = conv2(image)
image = bat(image)
image = relu(image)

print(image.shape)
