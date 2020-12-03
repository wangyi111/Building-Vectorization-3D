# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:15:43 2019

@author: davy_ks
"""

import torch

def RMSE(prediction, label):
    return torch.sqrt(torch.mean((prediction - label)**2)).item()

