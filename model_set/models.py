import torch
import scr.util as utils
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
from omegaconf import DictConfig


##### Original Unet ______  ####
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

class UNet(nn.Module):
    """Main UNet architecture """

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
        final = nn.functional.interpolate(selfFinal, input_data.size()[2:], mode='bilinear', align_corners=True)
        # print('Final shape after interpolating self.final(decode1), input_data.size()[2:]:', final.shape)
        
        return final

####   Unet Flood ___   ####

class DecodingBlockFlood(nn.Module):
    """Module in the decoding section of the UNet"""

    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlockFlood(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([output1, output2], 1))

class EncodingBlockFlood(nn.Module):
    """Convolutional batch norm block with relu activation (main block used in the encoding steps)
    The different in this version: padding = 0. As in the original paper, at eahc conv2D we loss a 
    2 pixel in H and W. 
    """

    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True, dropout=False, prob=0.5):
        super().__init__()

        if batch_norm:
            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      ]
        else:
            layers = [nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(), ]

        if dropout:
            layers.append(nn.Dropout(p=prob))

        self.EncodingBlock = nn.Sequential(*layers)
    
    def forward(self, input_data, fn=None):
        output = self.EncodingBlock(input_data)
        return output

class UNetFlood(nn.Module):
    """Main UNet architecture
    - This vertion is adapted to small input images, considering higher resolution DTM inputs. 
    NOTE: Flood context in genereal is well described by a short distance from the river ( max 1000m).
    This is an "IN PROGRESS" experiment. (Marz 21st 2023) Abdiel Fernandez RNCan
    @classes: Number of classes.
    @in_channels: Number of channels in the input image. 
    @classifierOn = whether compute linear steps with Sigmoid (True) or only convolution (False)
     NOTE: If classifierOn  Return 0-1 mask insted of logits. 
    """

    def __init__(self, classes, in_channels, dropout:bool = True, prob:float = 0.5, classifierOn = False):
        super().__init__()
        self.ClassifierOn = classifierOn
        self.conv1 = EncodingBlockFlood(in_channels, 64, dropout=dropout, prob=prob)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = EncodingBlockFlood(64, 128, dropout=dropout, prob=prob)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = EncodingBlockFlood(128, 256, dropout=dropout, prob=prob)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = EncodingBlockFlood(256, 512, dropout=dropout, prob=prob)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = EncodingBlockFlood(512, 1024, dropout=dropout, prob=prob)

        self.decode4 = DecodingBlockFlood(1024, 512)
        self.decode3 = DecodingBlockFlood(512, 256)
        self.decode2 = DecodingBlockFlood(256, 128)
        self.decode1 = DecodingBlockFlood(128, 64)
        self.final = nn.Conv2d(64, classes, kernel_size=1)
        self.linearChanelReduction = nn.Conv2d(classes,1, kernel_size=1)
        self.linear = nn.Conv2d(classes,classes, kernel_size=1)
        self.LRelu = nn.ReLU()
        self.output = nn.Sigmoid()

    def forward(self, input_data):
        conv1 = self.conv1(input_data)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        decode4 = self.decode4(conv4, center)
        decode3 = self.decode3(conv3, decode4)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)
        selfFinal = self.final(decode1)
        interpolation = nn.functional.interpolate(selfFinal, input_data.size()[2:], mode='bilinear', align_corners=True)
        linear1 = self.linear(interpolation)
        output = self.output(linear1)
        # if self.ClassifierOn:
            
        #     linear1Activated = self.LRelu(linear1)
        #     linear2 = self.linear(linear1Activated)
        #     linear2Activated = self.LRelu(linear2)
        #     linear3 = self.linearChanelReduction(linear2Activated)
        #     return self.LRelu(linear3)
        
        return output

####   UNet Classi Flood  ####

class EncodingBlock_LeakyRelu(nn.Module):
    '''
    Convolutional batch norm block with LeakyRelu(instead of Relu) activation (main block used in the encoding steps)
    The different in this version: padding = 0. As in the original paper, at eahc conv2D we loss a 
    2 pixel in H and W. 
    Differences vs original EncodingBlock:
     Since we do not have enough knowledge regarding flood modeling from DEM with CNN, we intend to       prevent the loss of some weights (zero values) in the convolution process.  
    
    '''
    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, 
                 batch_norm=True, dropout=False, prob=0.5, nSlope = 0.01):
        super().__init__()

        if batch_norm:
            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.LeakyReLU(negative_slope=nSlope),
                      nn.BatchNorm2d(out_size),
                      nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.LeakyReLU(negative_slope=nSlope),
                      nn.BatchNorm2d(out_size),
                      ]
        else:
            layers = [nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.LeakyReLU(negative_slope=nSlope),
                      nn.ReflectionPad2d(padding=0),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.LeakyReLU(negative_slope=nSlope),
                      ]

        if dropout:
            layers.append(nn.Dropout(p=prob))

        self.EncodingBlock = nn.Sequential(*layers)

    def forward(self, input_data, fn=None):
        output = self.EncodingBlock(input_data)
        return output

class DecodingBlock_LeakyRelu(nn.Module):
    """Module in the decoding section of the UNet"""

    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlock_LeakyRelu(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([output1, output2], 1))

class UNetClassiFlood(nn.Module):
    """Main UNet architecture This is an "in progress" experiment. (Marz 21st 2023)
    NOTE: Flood context in genereal is well described by a short distance from the river ( max 1000m).
    - This vertion is adapted to small input images, considering higher resolution DTM inputs. 
    - To avoid dead weights we change Relu by LeakyRelu and we play with negative slope value of LeakyRelu.    
    
    @classes: Number of classes.
    @in_channels: Number of channels in the input image. 
    """

    def __init__(self, classes, in_channels, dropout:bool = True, prob:float = 0.5,addParams:dict = None):
        super().__init__()
        self.addPrams = addParams
        self.NSlopeEncoder = addParams['negative_slope_Encoder'] 
        self.NSlopeLinear = addParams['negative_slope_linear']
        # self.Patch_W = addParams['patch_W']
        # self.Patch_W = addParams['patch_H']
        self.classes = classes
        self.conv1 = EncodingBlock_LeakyRelu(in_channels, 64, dropout=dropout, prob=prob)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = EncodingBlock_LeakyRelu(64, 128, dropout=dropout, prob=prob, nSlope=self.NSlopeEncoder)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = EncodingBlock_LeakyRelu(128, 256, dropout=dropout, prob=prob, nSlope=self.NSlopeEncoder)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = EncodingBlock_LeakyRelu(256, 512, dropout=dropout, prob=prob, nSlope=self.NSlopeEncoder)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = EncodingBlock_LeakyRelu(512, 1024, dropout=dropout, prob=prob, nSlope=self.NSlopeEncoder)
        self.decode4 = DecodingBlock_LeakyRelu(1024, 512)
        self.decode3 = DecodingBlock_LeakyRelu(512, 256)
        self.decode2 = DecodingBlock_LeakyRelu(256, 128)
        self.decode1 = DecodingBlock_LeakyRelu(128, 64)
        self.final2DConv = nn.Conv2d(64, classes, kernel_size=1)   
        self.linear1D = nn.Conv1d(1,1,kernel_size=1)
        # self.linear1DChanelReduction = nn.Conv1d(classes,1, kernel_size=1)
        self.maxpool_1D = nn.MaxPool1d(kernel_size=1)
        # self.leakyRelu = nn.LeakyReLU(negative_slope=self.NSlopeLinear)
        # self.output = nn.Sigmoid()
        
    def forward(self, input_data):
        conv1 = self.conv1(input_data)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        decode4 = self.decode4(conv4, center)
        decode3 = self.decode3(conv3, decode4)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)
        lastConv2D = self.final2DConv(decode1)  
        interpolation = nn.functional.interpolate(lastConv2D, input_data.size()[2:], mode='bilinear', align_corners=True)
        linear1Activated = F.leaky_relu(self.linear1D(interpolation.flatten(2)),negative_slope=self.NSlopeLinear)
        maxpool_1D = self.maxpool_1D(linear1Activated)
        linear2Activated = F.leaky_relu(self.linear1D(maxpool_1D,negative_slope=self.NSlopeLinear))       
        # linear2 = self.linear1D(maxpool_1D)
        # output = self.output(linear2.view(input_data.shape))
        return linear2Activated.view(input_data.shape)



