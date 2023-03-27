import torch
import scr.util as utils
from torch import nn
from torch import sigmoid


class EncodingBlock(nn.Module):
    """Convolutional batch norm block with relu activation (main block used in the encoding steps)"""

    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True,
                 dropout=False, prob=0.5):
        super().__init__()

        if batch_norm:
            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      ]
        else:
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(), ]

        if dropout:
            layers.append(nn.Dropout(p=prob))

        self.EncodingBlock = nn.Sequential(*layers)

    def forward(self, input_data):
        output = self.EncodingBlock(input_data)
        return output


class Interpolate(torch.nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class DecodingBlock(nn.Module):
    """Module in the decoding section of the UNet"""

    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlock(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([output1, output2], 1))
'''
class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

'''

class UNetFlood(nn.Module):
    """Main UNet architecture
    - This vertion is adapted to small input images, considering higher resolution DTM inputs. 
    NOTE: Flood context in genereal is well described by a short distance from the river ( max 400m).
    This is an "in progress" experiment. (Marz 21st 2023)
    """

    def __init__(self, classes, in_channels, dropout=False, prob=0.5):
        super().__init__()

        self.conv1 = EncodingBlock(in_channels, 64, dropout=dropout, prob=prob)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = EncodingBlock(64, 128, dropout=dropout, prob=prob)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = EncodingBlock(128, 256, dropout=dropout, prob=prob)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = EncodingBlock(256, 512, dropout=dropout, prob=prob)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = EncodingBlock(512, 1024, dropout=dropout, prob=prob)

        self.decode4 = DecodingBlock(1024, 512)
        self.decode3 = DecodingBlock(512, 256)
        self.decode2 = DecodingBlock(256, 128)
        self.decode1 = DecodingBlock(128, 64)

        self.final = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, input_data):
        # print('Encoding______')
        # print('input_data', input_data.shape)
        conv1 = self.conv1(input_data)
        # print('conv1', conv1.shape)
        maxpool1 = self.maxpool1(conv1)
        # print('maxpool1', maxpool1.shape)
        conv2 = self.conv2(maxpool1)
        # print('conv2', conv2.shape)
        maxpool2 = self.maxpool2(conv2)
        # print('maxpool2', maxpool2.shape)
        conv3 = self.conv3(maxpool2)
        # print('conv3', conv3.shape)
        maxpool3 = self.maxpool3(conv3)
        # print('maxpool3', maxpool3.shape)
        conv4 = self.conv4(maxpool3)
        # print('conv4', conv4.shape)
        maxpool4 = self.maxpool4(conv4)
        # print('maxpool4', maxpool4.shape)
        # print('center ______')
        center = self.center(maxpool4)
        # print('center', center.shape)
        # print('Decoding______')
        decode4 = self.decode4(conv4, center)
        # print('decode4 shape', decode4.shape)
        decode3 = self.decode3(conv3, decode4)
        # print('decode3 shape', decode3.shape)
        decode2 = self.decode2(conv2, decode3)
        # print('decode2 shape', decode2.shape)
        decode1 = self.decode1(conv1, decode2)
        # print('decode1 shape', decode1.shape)
        # print('setting final ___')
        selfFinal = self.final(decode1)
        # print('self.final(decode1) shape', selfFinal.shape)
        # print('input_data.size()[2:] shape', input_data.size()[2:])
        final = nn.functional.interpolate(self.final(decode1), input_data.size()[2:], mode='bilinear', align_corners=True)
        # print('Final shape after interpolating self.final(decode1), input_data.size()[2:]:', final.shape)
        
        return final

