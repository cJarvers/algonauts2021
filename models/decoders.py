'''Decoders, which take the ResNet outputs and return classifications, predictions, segmentations etc.'''

import torch
import torch.nn as nn

class EncoderDecoderPair(nn.Module):
    '''
    Container for an encoder and decoder pair.
    '''
    def __init__(self, encoder, decoder):
        super(EncoderDecoderPair, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_features = decoder.needs_features
        
    def forward(self, x):
        if self.use_features:
            features = self.encoder.features(x)
        else:
            features = self.encoder(x)
        y = self.decoder(features)
        return(y)


class ClassDecoder(nn.Module):
    '''
    Decoder for classification. Performs global average pooling, followed by a
    linear layer.
    
    Args:
        *num_classes (int): number of classes to map to
        *maps (int, default=2048): number of input maps
    '''
    needs_features = False
    
    def __init__(self, num_classes, maps=2048):
        super(ClassDecoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(maps, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Deconv2DDecoder(nn.Module):
    '''
    Decoder for semantic segmentation and similar pixel-level tasks.
    
    Args:
        *inplanes (int): number of feature maps in output of backbone
        *planes (list of int): number of feature maps to project to in each block
        *outplanes (list of int): number of feature maps each block should return
        *finallayer (torch.nn.Module): final layer
    
    Creates as many `Deconv2DBlock`s as there are `planes` and `outplanes`.
    Input x is first squeezed, then passed through each block in turn.
    Finally, `finallayer` is applied.
    '''
    needs_features = False
    
    def __init__(self, inplanes, planes, outplanes, finallayer):
        super(DeconvDecoder, self).__init__()
        self.blocks = []
        for p, o in zip(planes, outplanes):
            self.blocks.append(Deconv2DBlock(inplanes, p, o))
            inplanes = o
        self.finallayer = finallayer
        
    def forward(self, x):
        x = x.squeeze()
        for b in self.blocks:
            x = b(x)
        x = self.finallayer(x)
        return(x)

def make_cityscapes_decoder(backbone, weights=None):
    decoder = Deconv2DDecoder(2048, [512, 256, 128, 64, 64],
        [1024, 512, 256, 128, 64], torch.nn.Conv2d(64, 1, kernel_size=1))
    if weights is not None:
        decoder.load_state_dict(weights)
    return(decoder)
      
class UNet3DDecoder(nn.Module):
    '''
    Decoder for 3D semantic segmentation and similar pixel-level tasks.
    
    Args:
        *inplanes (list of int): number of input feature maps to each block
        *planes (list of int): number of feature maps to project to in each block
        *outplanes (list of int): number of feature maps each block should return
        *upsample (list of bool): for each block, indicates whether upsampling should occur
        *finallayer (torch.nn.Module): final layer
    
    Creates as many `Deconv3DBlock`s as there are `planes` and `outplanes`.
    Input xs should be a list of feature map tensors.
    The last feature is passed trough the first deconv block and the result is
    concatenated with the second-to-last feature. This is then passed through
    the second block and so on. The final output is passed through the `finallayer`.
    '''
    needs_features = True
    
    def __init__(self, inplanes, planes, outplanes, upsample, finallayer):
        super(UNet3DDecoder, self).__init__()
        self.blocks = []
        for i, p, o, u in zip(inplanes, planes, outplanes, upsample):
            self.blocks.append(Deconv3DBlock(i, p, o, upsample=u))
        self.finallayer = finallayer
        
    def forward(self, features):
        features.reverse()
        x = self.blocks[0](features[0])
        for b, f in zip(self.blocks[1:], features[1:]):
            x = b(torch.cat([f, x], dim=1))
        x = self.finallayer(x)
        return(x)
        
        
class Deconv2DBlock(nn.Module):
    '''
    Combines 2D convolutions, groupnorm, and upsampling.
    '''

    def __init__(self, inplanes, planes, outplanes, groups=32):
        super(Deconv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, groups=groups, padding=1)
        self.norm2 = nn.GroupNorm(groups, planes)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(groups, outplanes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.norm3(y)
        y = self.relu(y)
        y = self.upsample(y)
        return(y)

def identity(x):
    return x

class Deconv3DBlock(nn.Module):
    '''
    Combines 3D convolutions, groupnorm, and upsampling.
    '''

    def __init__(self, inplanes, planes, outplanes, groups=32, upsample=True):
        super(Deconv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, bias=False, groups=groups, padding=1)
        self.norm2 = nn.GroupNorm(groups, planes)
        self.conv3 = nn.Conv3d(planes, outplanes, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(groups, outplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)
        else:
            self.upsample = identity
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.norm3(y)
        y = self.relu(y)
        y = self.upsample(y)
        return(y)

