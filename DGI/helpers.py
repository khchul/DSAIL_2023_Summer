import torch
import torch.nn as nn
import torch.nn.functional as F

class corrupter():
    def __init__(self):

        return
    
    def corrupter(self, X, A):

        return 

class DGI(nn.Module):
    def __init__(self, xdim, adim, args):
        super(DGI, self).__init__()

        self.edim = args.edim

        if args.transductive == True:
            self.E_weights = nn.Linear()
        else:
            self.E_weights = nn.ModuleList(
                
            )

        self.Readout = lambda input : F.sigmoid(input.sum(dim=0))
        self.D_weights = nn.Linear(self.edim, self.edim)

    def forward(self, X, A):
        
        return