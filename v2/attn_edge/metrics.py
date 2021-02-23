# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:10:39 2019

@author: davy_ks
"""

import torch
SMOOTH = 1e-6

def RMSE(prediction, label):
    return torch.sqrt(torch.mean((prediction - label)**2)).item()

def MAE(prediction, label):
    return torch.mean(torch.abs(prediction-label)).item()

def NMAD(prediction, label):
    diff = torch.abs(prediction-label)
    median_d = torch.median(diff)
    median_m = torch.median(diff-median_d)
    return 1.4826 * median_m

def IoU(prediction,label):
    intersection = (prediction & label).float().sum((1,2))
    union = (prediction | label).float().sum((1,2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def Precision(prediction,label):
    tp = (prediction & label).float().sum((1,2,3))
    p = prediction.float().sum((1,2,3))
    return (tp+SMOOTH)/(p+SMOOTH)

def Recall(prediction,label):
    tp = (prediction & label).float().sum((1,2,3))
    #tpfn = tp + 
    pass
