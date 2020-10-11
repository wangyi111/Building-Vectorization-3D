# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:10:39 2019

@author: davy_ks
"""

import torch

def RMSE(prediction, label):
    return torch.sqrt(torch.mean((prediction - label)**2)).item()
