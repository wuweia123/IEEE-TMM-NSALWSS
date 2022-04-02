import torch.nn as nn
import torch.nn.functional as F
from network.densenet import densenet169
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.densenet169 = densenet169(pretrained=True)

        self.side3 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1),  nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
                                   nn.ReLU(inplace=True))
        self.side4 = nn.Sequential(nn.Conv2d(1280, 128, 3, padding=1),  nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
                                   nn.ReLU(inplace=True))
        self.side5 = nn.Sequential(nn.Conv2d(1664, 128, 3, padding=1),  nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
                                   nn.ReLU(inplace=True))
        self.side3catconv1 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.Conv2d(64, 1, 3, padding=1))
        self.side4catconv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.Conv2d(64, 1, 3, padding=1))
        self.side5catconv1 = nn.Conv2d(64, 1, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        self.block0 = nn.Sequential(self.densenet169.features.block0)
        self.block1 = nn.Sequential(self.densenet169.features.denseblock1)
        self.block2 = nn.Sequential(self.densenet169.features.transition1, self.densenet169.features.denseblock2)
        self.block3 = nn.Sequential(self.densenet169.features.transition2, self.densenet169.features.denseblock3)
        self.block4 = nn.Sequential(self.densenet169.features.transition3, self.densenet169.features.denseblock4)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.densenet169 = densenet169(pretrained=True)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        side2 = self.side3(x2)
        x3 = self.block3(x2)
        side3 = self.side4(x3)
        x4 = self.block4(x3)
        side4 = self.side5(x4)

        side4_up2 = self.upsample(side4)
        side4_up4 = self.upsample(side4_up2)
        side3_up2 = self.upsample(side3)


        side_3 = torch.cat((side4_up4, side3_up2, side2), 1)
        side_4 = torch.cat((side4_up2, side3), 1)
        side_5 = side4

        side_3 = self.side3catconv1(side_3)
        side_4 = self.side4catconv1(side_4)
        side_5 = self.side5catconv1(side_5)

        side_5 = F.interpolate(side_5, size=256, mode='bilinear', align_corners=None)
        side_4 = F.interpolate(side_4, size=256, mode='bilinear', align_corners=None)
        side_3 = F.interpolate(side_3, size=256, mode='bilinear', align_corners=None)

        side5_sig = self.sigmoid(side_5)
        side4_sig = self.sigmoid(side_4)
        side3_sig = self.sigmoid(side_3)

        side5_sig = torch.squeeze(side5_sig)
        side4_sig = torch.squeeze(side4_sig)
        side3_sig = torch.squeeze(side3_sig)


        return side3_sig, side4_sig, side5_sig#,gen_mid





