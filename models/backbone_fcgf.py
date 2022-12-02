import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, phase = 'encode'):
        print('Inside of forward method of FCGF')
        x = batch['features'].clone().detach()
        print('x.shape : ', x.shape)
        return x.to('cuda:0')