# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:53:22 2020

@author: davy_ks
"""

import torch
import torch.nn as nn
import numpy as np

import pdb

# source:
# https://github.com/xanderchf/MonoDepth-FPN-PyTorch/blob/master/main_fpn.py
class FullNormalLoss(nn.Module):
    def __init__(self):
        super(FullNormalLoss, self).__init__()

    def forward(self, input, target, mask=None):

        # TODO implement masked calculation
        if mask is not None:
            raise NotImplementedError

        grad_pred = self.imgrad_yx(input)
        grad_gt = self.imgrad_yx(target)

        #pdb.set_trace()       
    
        #prod = torch.mul(grad_pred[:, :, None, :],grad_gt[:, :, :, None])
        prod = torch.einsum('bijd,bikm->bijm', grad_pred[:, :, None, :], grad_gt[:, :, :, None])
        prod = prod.squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt(torch.sum(grad_pred**2, dim=-1))
        gt_norm = torch.sqrt(torch.sum(grad_gt**2, dim=-1))

        return 1 - torch.mean(prod / (pred_norm * gt_norm))

    def imgrad_yx(self, img):
        N, C, _, _ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

    def imgrad(self, img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        # grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
        return grad_y, grad_x
