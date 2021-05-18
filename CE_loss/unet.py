import torch
from torch import nn
from torch.utils import model_zoo
from blocks import BaseConv
from VGG16 import VGG16_backbone


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.vgg = VGG16_backbone()
        self.dmp = BackEnd()
        self.decoder = nn.Sequential(
            BaseConv(64, 128, 3, activation=nn.ReLU(),use_bn=False),
            BaseConv(128, 256, 3, activation=nn.ReLU(), use_bn=False),
            nn.Conv2d(in_channels = 256, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1),
        )

    def forward(self, input):

        input = self.vgg(input)
        conv2_2,conv3_3,conv4_3,conv5_3,x2,x3,x4=self.dmp(*input)
        dmp_out = self.decoder(x2)
        dmp_out = dmp_out/0.38

        return dmp_out

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)



    def forward(self, *input):
        conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)          # 1/16->1/8

        input = torch.cat([input, conv4_3], 1)  # 1/8:512+512=1024
        input = self.conv1(input)               # 1024->256
        x4 = self.conv2(input)               # 256->256

        input = self.upsample(x4)            # 1/8->1/4
        input = torch.cat([input, conv3_3], 1)  # 1/4:256+256=512
        input = self.conv3(input)               # 512->128
        x3 = self.conv4(input)               # 128->128

        input = self.upsample(x3)            # 1/4->1/2
        input = torch.cat([input, conv2_2], 1)  # 1/2: 128+128=256
        input = self.conv5(input)               # 256->64
        x2 = self.conv6(input)               # 64->64

        return conv2_2,conv3_3,conv4_3,conv5_3,x2,x3,x4


