import torch
import numpy as np
from torch import nn
import torchvision

class conv33_relu(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, outp, 3, bias=False, padding=1, ),
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class basic_block(nn.Module):
    def __init__(self, inp, outp, dropout_p=0.5):
        super().__init__()
        layers = []
        layers.append(conv33_relu(inp, outp))
        #if dropout_p > 0.0:
        #    layers.append(nn.Dropout2d(p=dropout_p))
        layers.append(conv33_relu(outp, outp))
        #if dropout_p > 0.0:
        #    layers.append(nn.Dropout2d(p=dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class contract_block(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.extract = basic_block(inp, outp)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feature = self.extract(x)
        x = self.maxpool(feature)
        return feature, x


class expansive_block(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        # convtranspos upsampling expand width and height , but reduce channel
        # bilinear upsampling only expand width and height.
        # if you want to use bilinear, you must reduce channel of x with addtional layer
        self.expansive = nn.ConvTranspose2d(inp, outp, kernel_size=2, stride=2)
        self.basic = basic_block(inp, outp)

    def forward(self, x, feature):
        # crop feature map into x size
        x = self.expansive(x)  # c * h * w -> c/2 * 2h * 2w

        centor_crop = torchvision.transforms.CenterCrop(x.size(2))
        crop_feature = centor_crop(feature)

        # channel wise concatenation
        new_ten = torch.cat([x, crop_feature], dim=1)

        return self.basic(new_ten)

class unet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.cont1 = contract_block(in_channel, 64)
        self.cont2 = contract_block(64, 128)
        self.cont3 = contract_block(128, 256)
        self.cont4 = contract_block(256, 512)

        self.basic = basic_block(512, 1024)

        self.exp4 = expansive_block(1024, 512)
        self.exp3 = expansive_block(512, 256)
        self.exp2 = expansive_block(256, 128)
        self.exp1 = expansive_block(128, 64)

        self.seg = nn.Conv2d(64, num_classes, 1)  # (Class, H, W)

        self.soft = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("input size : ", x.shape)
        feature1, x = self.cont1(x)
        # print("cont1 size: ", x.shape)
        feature2, x = self.cont2(x)
        # print("cont2 size: ", x.shape)
        feature3, x = self.cont3(x)
        # print("cont3 size: ", x.shape)
        feature4, x = self.cont4(x)
        # print("cont4 size: ", x.shape)

        x = self.basic(x)
        # print("basic size: ", x.shape)

        x = self.exp4(x, feature4)
        # print("exp4 size: ", x.shape)
        x = self.exp3(x, feature3)
        # print("exp3 size: ", x.shape)
        x = self.exp2(x, feature2)
        # print("exp2 size: ", x.shape)
        x = self.exp1(x, feature1)
        # print("exp1 size: ", x.shape)

        x = self.seg(x)
        # print("seg size: ", x.shape)

        x = self.sig(x)

        return x
    
class feature_extractor(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # Contracting path (downsampling)
        self.cont1 = contract_block(in_channel, 64)
        self.cont2 = contract_block(64, 128)
        self.cont3 = contract_block(128, 256)
        self.cont4 = contract_block(256, 512)

        self.basic = basic_block(512, 1024)

    def forward(self, x):
        # Feature extraction through contracting path
        feature1, x = self.cont1(x)
        feature2, x = self.cont2(x)
        feature3, x = self.cont3(x)
        feature4, x = self.cont4(x)

        x = self.basic(x)

        # Return features as a tuple
        features = (feature1, feature2, feature3, feature4)
        return x, features


class reconstructor(nn.Module):
    def __init__(self, out_channel):
        super().__init__()

        # Expansive path (upsampling)
        self.exp4 = expansive_block(1024, 512)
        self.exp3 = expansive_block(512, 256)
        self.exp2 = expansive_block(256, 128)
        self.exp1 = expansive_block(128, 64)

        # Segmentation output layer
        self.seg = nn.Conv2d(64, out_channel, 1)  # (Class, H, W)

        self.sig = nn.Sigmoid()  # Apply sigmoid to get probability map

    def forward(self, x, features):
        # Unpack features from the tuple
        feature1, feature2, feature3, feature4 = features

        # Expansive path (upsampling)
        x = self.exp4(x, feature4)
        x = self.exp3(x, feature3)
        x = self.exp2(x, feature2)
        x = self.exp1(x, feature1)

        # Final segmentation output
        x = self.seg(x)
        x = self.sig(x)

        return x






if __name__ == '__main__':
    model = unet(1, 4)
    a = torch.randn(1, 1, 240, 240)
    output = model(a)
    print(output)