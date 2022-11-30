import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch

class FCGF(nn.Module):

    def __init__(self, config):
        super(FCGF, self).__init__()

    def forward(self, batch):
        
        x = batch['features'].clone().detach()
        return torch.tensor(x).to('cuda:0')