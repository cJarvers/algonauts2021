'''Definition of ResNet50-3D model and auxiliary functions.'''
# Based on code from:
# 1. https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
#    commit: b2d4a80ad7ed669f33179a1928e85fb1e25012e5
#    Copyright (c) 2017 Kensho Hara, MIT License
#    License text: Permission is hereby granted, free of charge, to any person obtaining a copy 
#                  of this software and associated documentation files (the "Software"), to deal
#                  in the Software without restriction, including without limitation the rights
#                  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#                  copies of the Software, and to permit persons to whom the Software is
#                  furnished to do so, subject to the following conditions:
#
#                  The above copyright notice and this permission notice shall be included in all
#                  copies or substantial portions of the Software.
#
#                  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#                  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#                  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#                  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#                  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#                  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#                  SOFTWARE.
#
# 2. https://github.com/zhoubolei/moments_models/blob/v2/models.py
#    commit: 70e4855f5608c4481dfffd5f762e310d631d06c3
#    Copyright (c) 2018 MIT CSAIL and IBM Research, BSD 2-Clause License
#    License text: Redistribution and use in source and binary forms, with or without
#                  modification, are permitted provided that the following conditions are met:
#
#                  * Redistributions of source code must retain the above copyright notice, this
#                    list of conditions and the following disclaimer.
#
#                  * Redistributions in binary form must reproduce the above copyright notice,
#                    this list of conditions and the following disclaimer in the documentation
#                    and/or other materials provided with the distribution.
#
#                    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#                    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#                    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#                    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#                    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#                    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#                    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#                    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#                    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#                    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# For copyright of changes, see LICENSE file (MIT License).
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    '''Bottleneck block for a residual network with 3D convolutions.
    
    Args:
        *inplanes (int): number of input channels
        *planes (int): number of channels used within the block
        *groups (int, default 32): number of groups for GroupNorm
        *stride (int, default 1): stride for inner (3x3x3) convolution
        *downsample (default None): operation by which input is downsampled, if desired
        
    We make a few adjustments to the standard ResNet architecture:
    - We use GroupNorm (with 32 groups by default) instead of BatchNorm
    - In the 3x3-convolution we use groups (as many as in GroupNorm), reducing the
      number of free parameters
    - Use stride in the 3x3 convolutions, rather than 1x1, as in https://arxiv.org/abs/1512.03385
    '''
    expansion = 4
    
    def __init__(self, inplanes, planes, groups=32, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        # remember important parameters:
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        # construct sublayers:
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, bias=False,
            stride=stride, padding=1, groups=groups)
        self.norm2 = nn.GroupNorm(groups, planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(groups, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # apply layers
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.norm3(y)
        # apply residual
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x    
        y = self.relu(y + residual)
        return y


class ResNet3D50Backbone(nn.Module):
    '''
    3D ResNet50 backbone (convolutional / residual layers only, not fc layers).
    '''
    
    def __init__(self, blocktype=BottleneckBlock, layers=[3, 4, 6, 3], groups=32):
        super(ResNet3D50Backbone, self).__init__()
        # set up initial (non-residual) convolution
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.norm1 = nn.GroupNorm(groups, 64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # set up residual blocks
        self.inplanes = 64
        self.block1 = self._make_layer(blocktype, 64, layers[0], groups=groups)
        self.block2 = self._make_layer(blocktype, 128, layers[1], groups=groups, stride=2)
        self.block3 = self._make_layer(blocktype, 256, layers[2], groups=groups, stride=2)
        self.block4 = self._make_layer(blocktype, 512, layers[3], groups=groups, stride=2)
        

    def _make_layer(self, blocktype, planes, repetitions, groups=32, stride=1):
        # choose whether to downsample
        if stride != 1 or self.inplanes != planes * blocktype.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * blocktype.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(groups, planes * blocktype.expansion),
            )
        else:
            downsample = None
        # repeat the desired block for the required number of times
        layers = []
        layers.append(blocktype(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = planes * blocktype.expansion
        for _ in range(1, repetitions):
            layers.append(blocktype(self.inplanes, planes))
        return nn.Sequential(*layers)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
        
    def features(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)
        b1 = self.block1(x)
        b2 = self.block2(x)
        b3 = self.block3(x)
        b4 = self.block4(x)
        return([x, b1, b2, b3, b4])
        
