# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:54:41 2018

@author: davy_ks
"""

import torch
from torch import nn
from torch.nn import functional as F

import extractors
import pdb

import sys
#sys.path.append('/home/davy_ks/project/pytorch/GDLossGAN/')
#from torchviz import make_dot, make_dot_from_trace

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    #'resnet101': lambda: PSPNetWithSkipConnections(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        #priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        #p = F.upsample(input=x, size=(h, w), mode='bilinear')
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, output_nc=1, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=False): #pretrained=True at the beginning of the training 
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, output_nc, kernel_size=1),
            nn.Tanh()
            #nn.LogSoftmax()
        )

#        self.classifier = nn.Sequential(
#            nn.Linear(deep_features_size, 256),
#            nn.ReLU(),
#            nn.Linear(256, output_nc)
#        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)
        #print p.shape

        p = self.up_1(p)
        p = self.drop_2(p)
        #print p.shape

        p = self.up_2(p)
        p = self.drop_2(p)
        #print p.shape

        p = self.up_3(p)
        p = self.drop_2(p)
        #print p.shape
        
        #print self.final(p).shape

        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

#        # Network visualization         
#        make_dot(self.final(p)).view()


        #return self.final(p), self.classifier(auxiliary)
        return self.final(p)
        
## Add long skip connections
class PSPNetWithSkipConnections(nn.Module):
    def __init__(self, output_nc=1, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super(PSPNetWithSkipConnections, self).__init__()
               
        
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256) 
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        
        self.fconv_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fconv_2 = nn.Conv2d(128, 64, kernel_size=1)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, output_nc, kernel_size=1),
            nn.Tanh() ## I added
            #nn.LogSoftmax()
        )

#        self.classifier = nn.Sequential(
#            nn.Linear(deep_features_size, 256),
#            nn.ReLU(),
#            nn.Linear(256, output_nc)
#        )

    def forward(self, x):
        

        skip_1, skip_2, class_f, f = self.feats(x) 
        
        p = self.psp(f)
        p = self.drop_1(p)
        #print p.shape
       
        p = self.up_1(p)
        p = torch.cat((p,skip_1), 1)
        p = self.fconv_1(p)
        p = self.drop_2(p)
        #print p.shape
        
        p = self.up_2(p)

        p = torch.cat((p,skip_2), 1)
        p = self.fconv_2(p)
        p = self.drop_2(p)
        #print p.shape

        p = self.up_3(p)
        p = self.drop_2(p)
        #print p.shape
        
        #print self.final(p).shape

        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        # Network visualization         
#        make_dot(self.final(p)).view()
#        quit()

        #return self.final(p), self.classifier(auxiliary)
        return self.final(p)
