import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch, phase = 'encode'):
        x = batch['features'].clone().detach()
        return x.to('cuda:0')