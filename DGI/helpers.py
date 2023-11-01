import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from torch_geometric.utils import dropout_edge

class Corrupter():
    def __init__(self):
        self.init_crpt()

    def init_crpt(self):
        self.feat_crpt = lambda X: X[torch.randperm(X.size(0))]
        self.adj_crpt = lambda A, p:dropout_edge(edge_index=A, p=p, force_undirected=True)

    def corrupt(self, type, batch, p=.5):
        crpt_batch = {
            'x': batch['x'],
            'edge_index': batch['edge_index']
        }

        if type == 'feature' or type == 'both':
            crpt_batch['x'] = self.feat_crpt(batch['x'])
        elif type == 'adjacency' or type == 'both':
            crpt_batch['edge_index'] = self.adj_crpt(batch['edge_index'], p)

        return crpt_batch
            


class DGI(nn.Module):
    def __init__(self, F_dim, E_dim, type):
        super(DGI, self).__init__()

        self.F_dim = F_dim
        self.E_dim = E_dim
        self.type = type

        if type == 'transductive':
            self.GCN = gnn.Sequential('x, edge_index',[
                (gnn.GCNConv(self.F_dim, self.E_dim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True)
            ]
            )
        else:
            # Three-layer mean pooling
            self.GCN = gnn.Sequential('x, edge_index',[
                (gnn.SAGEConv((self.F_dim, self.F_dim), self.E_dim), 'x, edge_index -> x'),
                nn.PReLU(),
                (gnn.SAGEConv((self.E_dim, self.E_dim), self.E_dim), 'x, edge_index -> x'),
                nn.PReLU(),
                (gnn.SAGEConv((self.E_dim, self.E_dim), self.E_dim), 'x, edge_index -> x'),
                nn.PReLU()
                ]
            )
            #raise NotImplementedError

        self.readout = lambda input : F.sigmoid(input.mean(dim=0))
        self.W = nn.Linear(self.E_dim, self.E_dim)

    def encode(self, X, A):
        return self.GCN(X, A)
        
    def discriminate(self, H, s):
        return torch.matmul(H, self.W(s))
