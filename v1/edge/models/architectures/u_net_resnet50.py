import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
import pdb

def get_weight_bilinear(num_channel_in, num_channel_out, size_kernel):
    factor = (size_kernel + 1) // 2
    #pdb.set_trace()
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


class UNetResNet50(torch.nn.Module):
    """
    DIFFERENCES WITH OFFICIAL U-NET IMPLEMENTATION
    - used Batch Normalization in decoder blocks (Conv2d + BatchNorm + ReLU)
    """
    def __init__(self, num_class, is_pretrained=False):
        super(UNetResNet50, self).__init__()
        # make encoder
        # import and extract convolutional layers, ignore bottleneck layers
        module_resnet50 = resnet50(pretrained=is_pretrained)
        pdb.set_trace()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).apply(self.initialize)
        self.module_encoder_0 = torch.nn.Sequential(
            module_resnet50.bn1,
            module_resnet50.relu
        )
        self.module_encoder_0_maxpool = module_resnet50.maxpool
        self.module_encoder_1 = module_resnet50.layer1
        self.module_encoder_2 = module_resnet50.layer2
        self.module_encoder_3 = module_resnet50.layer3
        self.module_encoder_4 = module_resnet50.layer4
        # make encoder-to-decoder skip-connections
        self.module_encoder_0_skip = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=1
        )
        torch.nn.init.constant_(self.module_encoder_0_skip.weight, 0)
        torch.nn.init.constant_(self.module_encoder_0_skip.bias, 0)
        self.module_encoder_1_skip = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=1
        )
        torch.nn.init.constant_(self.module_encoder_1_skip.weight, 0)
        torch.nn.init.constant_(self.module_encoder_1_skip.bias, 0)
        self.module_encoder_2_skip = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=1
        )
        torch.nn.init.constant_(self.module_encoder_2_skip.weight, 0)
        torch.nn.init.constant_(self.module_encoder_2_skip.bias, 0)
        self.module_encoder_3_skip = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=1
        )
        torch.nn.init.constant_(self.module_encoder_3_skip.weight, 0)
        torch.nn.init.constant_(self.module_encoder_3_skip.bias, 0)
        # make decoder
        # blocks 1, 2 and 3: 2x up-sampling
        # block 0: 4x up-sampling
        # weights initialized as bilinear up-sampling filters
        #pdb.set_trace()
        self.module_decoder_3_upconv = torch.nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ).apply(self.initialize)
        self.module_decoder_3_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        ).apply(self.initialize)
        self.module_decoder_2_upconv = torch.nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ).apply(self.initialize)
        self.module_decoder_2_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        ).apply(self.initialize)
        self.module_decoder_1_upconv = torch.nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ).apply(self.initialize)
        self.module_decoder_1_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        ).apply(self.initialize)
        self.module_decoder_0_upconv = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ).apply(self.initialize)
        self.module_decoder_0_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        ).apply(self.initialize)
        # make classifier
        self.module_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=num_class,
                kernel_size=1
            ),
            torch.nn.Tanh()
        ).apply(self.initialize)
            
        #torch.nn.init.constant_(self.module_classifier.weight, 0)
        #torch.nn.init.constant_(self.module_classifier.bias, 0)

    @staticmethod
    def initialize(module_layer):
        if type(module_layer) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(module_layer.weight)
            if module_layer.bias is not None:
                torch.nn.init.constant_(module_layer.bias, 0)
        if type(module_layer) == torch.nn.ConvTranspose2d:
            tensor_weight = get_weight_bilinear(
                num_channel_in=module_layer.in_channels,
                num_channel_out=module_layer.out_channels,
                size_kernel=module_layer.kernel_size[0]
            )
            module_layer.weight.data.copy_(tensor_weight)
            # torch.nn.init.kaiming_uniform_(module_layer.weight)
            if module_layer.bias is not None:
                torch.nn.init.constant_(module_layer.bias, 0)
                
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                

    def forward(self, tensor_in):
        # encoder
        tensor_in = self.conv1(tensor_in)
        tensor_out = self.module_encoder_0(tensor_in)
        # output size = input_size / 2
        tensor_encoder_skip_0 = self.module_encoder_0_skip(tensor_out)
        tensor_out = self.module_encoder_0_maxpool(tensor_out)
        # output size = input_size / 4
        tensor_out = self.module_encoder_1(tensor_out)
        tensor_encoder_skip_1 = self.module_encoder_1_skip(tensor_out)
        tensor_out = self.module_encoder_2(tensor_out)
        # output size = input_size / 8
        tensor_encoder_skip_2 = self.module_encoder_2_skip(tensor_out)
        tensor_out = self.module_encoder_3(tensor_out)
        # output size = input_size / 16
        tensor_encoder_skip_3 = self.module_encoder_3_skip(tensor_out)
        tensor_out = self.module_encoder_4(tensor_out)
        # output size = input_size / 32
        # decoder
        tensor_out = self.module_decoder_3_upconv(tensor_out)
        tensor_out = self.module_decoder_3_conv(
            torch.cat([tensor_out, tensor_encoder_skip_3], dim=1)
        )
        # output size = input_size / 16
        tensor_out = self.module_decoder_2_upconv(tensor_out)
        tensor_out = self.module_decoder_2_conv(
            torch.cat([tensor_out, tensor_encoder_skip_2], dim=1)
        )
        # output size = input_size / 8
        tensor_out = self.module_decoder_1_upconv(tensor_out)
        tensor_out = self.module_decoder_1_conv(
            torch.cat([tensor_out, tensor_encoder_skip_1], dim=1)
        )
        # output size = input_size / 4
        tensor_out = self.module_decoder_0_upconv(tensor_out)
        tensor_out = self.module_decoder_0_conv(
            torch.cat([tensor_out, tensor_encoder_skip_0], dim=1)
        )
        # output size = input_size / 2
        tensor_out = torch.nn.functional.interpolate(
            tensor_out,
            scale_factor=2.0,
            mode="bilinear"
        )
        # output size = input_size
        # classifier
        tensor_out = self.module_classifier(tensor_out)

        return tensor_out
