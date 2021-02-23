# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:03:45 2020

@author: davy_ks
"""

import argparse
import numpy as np
import random
from PIL import Image
import xdibias
import pdb 
import matplotlib.pyplot as plt
import random
from scipy.stats import truncnorm
from scipy.ndimage import gaussian_filter

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas

def outlier_mask(img, length = 100):  # In the first test (name_v2) the was length = 50
    
    canvas = np.ones((img.shape[0], img.shape[1])).astype("i")
    ini_x = random.randint(0, img.shape[0] - 1)
    ini_y = random.randint(0, img.shape[1] - 1)
    mask = random_walk(canvas, ini_x, ini_y, length ** 2)
    
    
    return mask

def outlier_simulation(img, low=0.0, high=1.0):
    mask = outlier_mask(img)
    #pdb.set_trace()

    img[mask == 0] = gaussian_filter(np.random.uniform(low,high),sigma=1)
    
    #plt.show(plt.imshow(img))
    
    return img