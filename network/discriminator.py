"designed by Wuwei"
import torch.nn as nn


class Dis(nn.Module):
    def __init__(self, num_classes=1, ndf = 32):
        super(Dis, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*8, ndf*4, kernel_size=3, stride=1, padding=1)
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*4, ndf*2, kernel_size=3, stride=1, padding=1)
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ndf*2, ndf*1, kernel_size=3, stride=1, padding=1)
        )

        self.classifier = nn.Conv2d(ndf, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.deconv4(x)
        x = self.leaky_relu(x)
        x = self.deconv3(x)
        x = self.leaky_relu(x)
        x = self.deconv2(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        x = self.sigmoid(x)
        return x

