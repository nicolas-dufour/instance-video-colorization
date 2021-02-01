from collections import OrderedDict
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32):
        super(UNet, self).__init__()

        self.encoder1 = UNet._conv_block(in_channels, features, name='encoder1')
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._conv_block(features, 2*features, name='encoder2')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._conv_block(2*features, 4*features, name='encoder3')
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._conv_block(4*features, 8*features, name='encoder4')
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.embedding = UNet._conv_block(8*features, 16*features, name='embedding')

        self.upconv4 = UNet._up_conv(16*features, 8*features, name='upconv4')
        self.decoder4 = UNet._conv_block(16*features, 8*features, name='decoder4')

        self.upconv3 = UNet._up_conv(8*features, 4*features, name='upconv3')
        self.decoder3 = UNet._conv_block(8*features, 4*features, name='decoder3')

        self.upconv2 = UNet._up_conv(4*features, 2*features, name='upconv2')
        self.decoder2 = UNet._conv_block(4*features, 2*features, name='decoder2')

        self.upconv1 = UNet._up_conv(2*features, features, name='upconv1')
        self.decoder1 = UNet._conv_block(2*features, features, name='decoder1')

        self.out_conv = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.maxpool1(enc1))

        enc3 = self.encoder3(self.maxpool2(enc2))

        enc4 = self.encoder4(self.maxpool3(enc3))

        embedding = self.embedding(self.maxpool4(enc4))

        up4 = self.upconv4(embedding)
        up4_skip = torch.cat((up4, enc4), dim=1)
        dec4 = self.decoder4(up4_skip)

        up3 = self.upconv3(dec4)
        up3_skip = torch.cat((up3, enc3), dim=1)
        dec3 = self.decoder3(up3_skip)

        up2 = self.upconv2(dec3)
        up2_skip = torch.cat((up2, enc2), dim=1)
        dec2 = self.decoder2(up2_skip)

        up1 = self.upconv1(dec2)
        up1_skip = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(up1_skip)

        out = self.out_conv(dec1)

        return out

    @staticmethod
    def _conv_block(input_dim, output_dim, name):
        return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name+"conv1",
                            nn.Conv2d(
                                in_channels=input_dim,
                                out_channels=output_dim,
                                kernel_size=3,
                                padding=1,
                                bias=False
                            )
                        ),
                        (
                            name+'relu1',
                            nn.ReLU(inplace=True)
                        ),
                        (
                            name+"conv2",
                            nn.Conv2d(
                                in_channels=output_dim,
                                out_channels=output_dim,
                                kernel_size=3,
                                padding=1,
                                bias=False
                            )
                        ),
                        (
                            name+'relu12',
                            nn.ReLU(inplace=True)
                        )
                    ]
                )
        )

    @staticmethod
    def _up_conv(input_dim, output_dim, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name+'upsample',
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=True
                        )
                    ),
                    (
                        name+'conv',
                        nn.Conv2d(
                            in_channels=input_dim,
                            out_channels=output_dim,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        )
                    )
                ]
            )
        )