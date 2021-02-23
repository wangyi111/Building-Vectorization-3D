import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d




class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, multi_scale_upsample = True):




        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception_dense_feature':
            if multi_scale_upsample:
                low_level_inplanes = [64,128,256,728]  ##1/2, 1/4 1/8 1/16
            else:
                low_level_inplanes = [128] 
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError


   #### 1X1 conv to reduce num of channels for skip connection
        #conv for 1/2 features
        self.skip1_2 = nn.Sequential(
                nn.Conv2d(low_level_inplanes[0], 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU())


        #conv for 1/4 features
        self.skip1_4 = nn.Sequential(
                nn.Conv2d(low_level_inplanes[1], 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU())

        #conv for 1/8 features
        self.skip1_8 = nn.Sequential(
                nn.Conv2d(low_level_inplanes[2], 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU())

        #conv for 1/16 features
        self.skip1_16 = nn.Sequential(
                nn.Conv2d(low_level_inplanes[3], 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU())        
        
        self.skip = [self.skip1_2, self.skip1_4, self.skip1_8, self.skip1_16]


        #conv for 1/16 features (after aspp, 256, plus a 48 dim feature) 
        self.aspp_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1))


        ####does not use depthwise separable conv here, not necessarily improve performance, just reduce computational burden 
        ##### before upsampling
        self.last_conv2 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
                                        
        self.last_conv4 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1))
               
        self.last_conv8 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1))

        self.last_convs = [self.last_conv2, self.last_conv4, self.last_conv8, self.aspp_conv]

        self._init_weight()



    def forward(self, x, low_level_feat):#x, is output after aspp (256 channel), low_level_feat is a list of low level features from entryflow conv2, block 1, 3, 13 (1/2, 1/4, 1/8, 1/16)

      
        
        
        low_level_feat_reduced = []    ####skip feature after convolution with reduced num of channels
        for i,j in enumerate(low_level_feat):
            low_level_feat_reduced.append(self.skip[i](j))
            
            

        for i in range(len(self.last_convs)):
            x = torch.cat((x, low_level_feat_reduced[-i-1]), dim=1)
            #x = F.interpolate(x, size=low_level_feat[-i-1].size()[2:]*2, mode='bilinear', align_corners=True)
            ### Change here. not flexible if size not devisible
            self.size=low_level_feat[-i-1].size()[2:]
            x = F.interpolate(x, size=tuple(I*2 for I in self.size), mode='bilinear', align_corners=True)
            x = self.last_convs[-i-1](x)

        ##tensor.size()*2 returns a 2*length

        return x



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
