'''Decoders, which take the ResNet outputs and return classifications, predictions, segmentations etc.'''

import torch
import torch.nn as nn

class ClassDecoder(nn.Module):
    '''
    Decoder for classification. Performs global average pooling, followed by a
    linear layer.
    
    Args:
        *num_classes (int): number of classes to map to
        *maps (int, default=2048): number of input maps
    '''
    
    def __init__(self, num_classes, maps=2048):
        super(ClassDecoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(maps, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
