import torch
import numpy as np
from torch import nn
import torchvision
import torch.nn.functional as F

class conv33_relu(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, outp, 3, bias=False, padding=1, ),
            nn.GroupNorm(16, outp),
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
        original_size = x.shape[2:]
        feature = self.extract(x)
        feature = self.maxpool(feature)
        return feature, original_size


class expansive_block(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        # convtranspos upsampling expand width and height , but reduce channel
        # bilinear upsampling only expand width and height.
        # if you want to use bilinear, you must reduce channel of x with addtional layer
        self.expansive = nn.ConvTranspose2d(inp, outp, kernel_size=2, stride=2)
        self.basic = basic_block(outp, outp)

    def forward(self, x, original_size):
        x = self.expansive(x) 
        # original_size에 x를 맞춤
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        x = self.basic(x)
        return x



        

class encoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.cont1 = contract_block(in_channel, 64)
        self.cont2 = contract_block(64, 128)
        self.cont3 = contract_block(128, 256)
        self.basic = basic_block(256, 512)

    def forward(self, x):
        x, size1 = self.cont1(x)
        x, size2 = self.cont2(x)
        x, size3 = self.cont3(x)
        x = self.basic(x)
        size = [size1, size2, size3]
        

        return x, size
        
        
class decoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.exp3 = expansive_block(512, 256)
        self.exp2 = expansive_block(256, 128)
        self.exp1 = expansive_block(128, 64)
        self.seg = nn.Conv2d(64, out_channel, 1)

        self.sig = nn.Sigmoid()


    def forward(self, x, size):
        x = self.exp3(x, size[2])
        x = self.exp2(x, size[1])
        x = self.exp1(x, size[0])
        x = self.seg(x)

        return self.sig(x)
    

if __name__ == '__main__':
    x = torch.rand(1,3,300,400)

    enc = encoder(in_channel=3)
    latent, size = enc(x)
    print(latent.shape)
    dec = decoder(out_channel=3)
    output = dec(latent, size)
    print(output.shape)
    
