# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:48:03 2019

@author: davy_ks
"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            #print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model, cur_iter, epoch):
        '''Saves model when validation loss decrease.'''
        
        state = {
            "iteration": cur_iter,
            "epoch": epoch,
            "modelG_states": model.concat_netG.state_dict(),
            "modelD_states": model.netD.state_dict(),
            "optimizer_G": model.optimizer_G.state_dict(),
            "optimizer_D": model.optimizer_D.state_dict()
        }
        if self.verbose:
            print('Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        torch.save(state, model.save_dir + "/best.pkl")

        self.val_score_min = val_score