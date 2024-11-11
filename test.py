import torch.nn.functional as F
import torch

image = torch.rand((3,128,15))
print(image.shape)
image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
print(image.shape)
