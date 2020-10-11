# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:38:34 2020

@author: davy_ks
"""

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#   
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
###############################################################################
'''
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Introduced partial convolutoins based padding for convolutional layers
'''

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from partialconv2d import PartialConv2d
import numpy as np

import pdb

__all__ = ['PConvResNet', 'pcresnet18', 'pcresnet34', 'pcresnet50', 'pcresnet101',
           'pcresnet152']


# model_urls = {
#     'pcresnet18': '',
#     'pcresnet34': '',
#     'pcresnet50': '',
#     'pcresnet101': '',
#     'pcresnet152': '',
# }

model_urls = {
    'pcresnet18': '',
    'pcresnet34': '',
    'pcresnet50': '',
    'pcresnet101': '',
    'pcresnet152': '',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, multi_channel=True, return_mask=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        
        pdb.set_trace()
        
        x = input[0]
        mask = input[1]
        
        residual = x

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, mask


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_conv=None, downsample_bn=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, multi_channel=True, return_mask=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, multi_channel=True, return_mask=True)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = downsample_conv
        self.downsample_bn = downsample_bn
        self.stride = stride

    def forward(self, input):
        
        x = input[0]        
        mask = input[1]
               
        residual = x

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, mask = self.conv3(out, mask)
        out = self.bn3(out)

        if self.downsample_conv is not None:

            residual, _ = self.downsample_conv(input[0], input[1])
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)

        return out, mask


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output
    
class PConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(PConvResNet, self).__init__()
        self.conv1 = PartialConv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample_conv = None
        downsample_bn = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
#            downsample = nn.Sequential(
#                PartialConv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False, multi_channel=True, return_mask=True),
#                nn.BatchNorm2d(planes * block.expansion),
#            )
            downsample_conv = PartialConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, multi_channel=True, return_mask=True)
            downsample_bn = nn.BatchNorm2d(planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_conv, downsample_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask):
        
        
        x, mask = self.conv1(x, mask)
        x = self.bn1(x)
        x = self.relu(x)
        x, indices = self.maxpool(x)
        
        # retrieve corresponding elements from data2 according to indices
        mask = retrieve_elements_from_indices(mask, indices)
       
        pdb.set_trace()
               
        x, mask = self.layer1([x,mask])
        x, mask = self.layer2([x,mask])
        x, mask = self.layer3([x,mask])
        x, mask = self.layer4([x,mask])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def pcresnet18(pretrained=False, **kwargs):
    """Constructs a PConvResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PConvResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pcresnet18']))
    return model


def pcresnet34(pretrained=False, **kwargs):
    """Constructs a PConvResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PConvResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pcresnet34']))
    return model


def pcresnet50(pretrained=False, **kwargs):
    """Constructs a PConvResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PConvResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pcresnet50']))
    return model


def pcresnet101(pretrained=False, **kwargs):
    """Constructs a PConvResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PConvResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pcresnet101']))
    return model


def pcresnet152(pretrained=False, **kwargs):
    """Constructs a PConvResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PConvResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pcresnet152']))
    return model

#################################################################################################################
    
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, multi_channel=True, return_mask=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity
        
        #self._init_weight()

                
    def forward(self, input):
        x = input[0]
        mask = input[1]
        
        x, mask = self.conv(x, mask)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x, mask
        
class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, mask):
        return self.bridge([x, mask])
        
        
class UpBlockForUNetWithResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="nearest"): # initial was "conv_transpose"
        super(UpBlockForUNetWithResNet, self).__init__()
        
        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.upsample = nn.Upsample(mode='nearest', scale_factor=2)
            #nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) ## original
            #nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)                
        self.pconv = PartialConv2d(up_conv_in_channels, up_conv_out_channels, padding=padding, kernel_size=1, stride=1, multi_channel=True, return_mask=True)
            
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

        #self._init_weight()        
                
    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        #pcb.set_trace()
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

    def get_weight_bilinear(self, num_channel_in, num_channel_out, size_kernel):
        factor = (size_kernel + 1) // 2
        #pcb.set_trace()
        if size_kernel % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size_kernel, :size_kernel]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)[np.newaxis,np.newaxis,:,:]
        weight = np.ones(
            (num_channel_in, num_channel_out, size_kernel, size_kernel),
            dtype=np.float64
        )
        '''weight[
            list(range(num_channel_in)),
            list(range(num_channel_out)),
            :,
            :
        ] = filt'''
        weight = np.matmul(weight,filt)
        return torch.from_numpy(weight).float()

class PConvUNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=1):
        super(PConvUNetWithResnet50Encoder, self).__init__()
        resnet = pcresnet50(pretrained=False)
        down_blocks = []
        up_blocks = []
                
        self.input_conv = PartialConv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, multi_channel=True, return_mask=True)
                               
        # if input image has 1 channel
        self.input_block = nn.Sequential(*list(resnet.children()))[1:3]
        
        #self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 1, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
                                                    
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        
        self.tanh = nn.Tanh()
        
        self.__init_weight()

#    @staticmethod
#    def initialize(layer):
#        firstlayer_weight = nn.Sequential(*list(resnet.children()))[0].weight
#        onechannel_weight = firstlayer_weight[:,1,:,:].unsqueeze_(1)
#        layer.weight.data.copy_(onechannel_weight)

    def forward(self, x, mask, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x, mask = self.input_conv(x, mask)
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x, indices = self.input_pool(x)

        # retrieve corresponding elements from data2 according to indices
        mask = retrieve_elements_from_indices(mask, indices)

        
        
        for i, block in enumerate(self.down_blocks, 2):
            x, mask = block([x,mask])
            if i == (PConvUNetWithResnet50Encoder.DEPTH - 1):
                continue
            #pre_pools["layer_{i}"] = x
            pre_pools["layer_{0}".format(str(i))] = x

       
        
        x, mask = self.bridge(x, mask)
        
        pdb.set_trace()

        for i, block in enumerate(self.up_blocks, 1):
            key = "layer_{0}".format(str(PConvUNetWithResnet50Encoder.DEPTH - 1 - i))
            #key = "layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        x = self.out(x)
        del pre_pools

        x = self.tanh(x)
        
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
                
class pcCoupledUNetupperResnet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=1, backbone = "resnet50"):
        super(pcCoupledUNetupperResnet, self).__init__()
        
        if backbone == "resnet50":
            resnet = pcresnet50(pretrained=False)
        elif backbone == "resnet101":
            resnet = pcresnet101(pretrained=False)
        elif backbone == "resnet152":
            resnet = pcresnet152(pretrained=False)
        
        dsm_down_blocks = []
        img_down_blocks = []
        
        up_blocks = []
        
        ################# DSM Branch #############################################
        self.dsm_input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        # if input image has 1 channel
        self.dsm_input_block = nn.Sequential(*list(resnet.children()))[1:3]
        
        #self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        
        self.dsm_input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                dsm_down_blocks.append(bottleneck)
        self.dsm_down_blocks = nn.ModuleList(dsm_down_blocks)
        
        ## IMG Branch        
        self.img_input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        # if input image has 1 channel
        self.img_input_block = nn.Sequential(*list(resnet.children()))[1:3]
        
        #self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        
        self.img_input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                img_down_blocks.append(bottleneck)
        self.img_down_blocks = nn.ModuleList(img_down_blocks)


        ################# Resize Skip connections ###############################################
        
        self.downsize_pools = nn.ModuleList( [nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False),
                                                               nn.BatchNorm2d(64)),
                                              nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
                                                               nn.BatchNorm2d(256)), 
                                              nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
                                                               nn.BatchNorm2d(512) ),
                                              nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),
                                                               nn.BatchNorm2d(1024) ) ])
                                                                                                                 
                                                                                                                            
        ################# DECODER ###############################################
        self.prebridge = nn.Conv2d(4096, 2048, kernel_size=1, stride=1)
        self.bridge = Bridge(2048, 2048)
        
        up_blocks.append(UpBlockForUNetWithResNet(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 2, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
                                                    
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        
        self.tanh = nn.Tanh()
        
        self.__init_weight()

#    @staticmethod
#    def initialize(layer):
#        firstlayer_weight = nn.Sequential(*list(resnet.children()))[0].weight
#        onechannel_weight = firstlayer_weight[:,1,:,:].unsqueeze_(1)
#        layer.weight.data.copy_(onechannel_weight)

        
                
    def forward(self, *input):
        #pcb.set_trace()
    
        ####################  DSM Branch  ###############################
        dsm_pre_pools = dict()
        dsm_pre_pools["dsm_layer_0"] = input[0]

        dsm = self.dsm_input_conv(input[0])
        dsm = self.dsm_input_block(dsm)
        dsm_pre_pools["dsm_layer_1"] = dsm
        dsm = self.dsm_input_pool(dsm)

        
        for i, block in enumerate(self.dsm_down_blocks, 2):
            dsm = block(dsm)
            if i == (pcCoupledUNetupperResnet.DEPTH - 1):
                continue
            dsm_pre_pools["dsm_layer_{0}".format(str(i))] = dsm

        ####################  IMG Branch  ###############################
        img_pre_pools = dict()
        img_pre_pools["img_layer_0"] = input[1]

        img = self.img_input_conv(input[1])
        img = self.img_input_block(img)
        img_pre_pools["img_layer_1"] = img
        img = self.img_input_pool(img)

        for i, block in enumerate(self.img_down_blocks, 2):
            img = block(img)
            if i == (pcCoupledUNetupperResnet.DEPTH - 1):
                continue
            img_pre_pools["img_layer_{0}".format(str(i))] = img
            
            
        #########  Reduce size of skip connections #####################         
        
        downsizepools = dict()

        for pool in range(1, len(img_pre_pools)):
            key = "layer_{0}".format(str(pool))
            downsizepools[key] = self.downsize_pools[pool-1]
        
            
        
        ########################  UPSAMPLING ###########################
        
        fusion = self.prebridge(torch.cat((dsm,img), 1))   
        x = self.bridge(fusion)

        
        for i, block in enumerate(self.up_blocks, 1):
            dsm_key = "dsm_layer_{0}".format(str(pcCoupledUNetupperResnet.DEPTH - 1 - i))
            img_key = "img_layer_{0}".format(str(pcCoupledUNetupperResnet.DEPTH - 1 - i))
            key = "layer_{0}".format(str(pcCoupledUNetupperResnet.DEPTH - 1 - i))
    
            dsm_pre_pool = dsm_pre_pools[dsm_key]
            img_pre_pool = img_pre_pools[img_key]

            if key in downsizepools.keys():
                x = block(x, downsizepools[key](torch.cat((dsm_pre_pool,img_pre_pool),1)))
            else:
                x = block(x, torch.cat((dsm_pre_pool,img_pre_pool),1))
            
            del dsm_pre_pool, img_pre_pool

        x = self.out(x)

        x = self.tanh(x)

        del dsm_pre_pools, img_pre_pools  
        
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
                
                
                
### Version 3 - Test "CoupledUNetResnet"
model = PConvUNetWithResnet50Encoder()
#print model
model.cuda()
inp1 = torch.rand((2, 1, 256, 256)).cuda()
mask = torch.ones((2, 1, 256, 256)).cuda()
mask[:,:,10:20,30:60] = 0
out = model(inp1, mask)

#inp = []
#inp.append(inp1)
#inp.append(inp2)
#
#out = model(*inp)