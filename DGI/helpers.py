import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
import torch_geometric.utils as gnu

class Corrupter():
    def __init__(self, N_graph):
        self.N_graph = N_graph
        self.init_crpt()

    def init_crpt(self):
        feat_crpt_fns = []
        adj_crpt_fns = []

        for _ in range(self.N_graph):
            feat_crpt_fns.append(lambda X: X[torch.randperm(X.size(0))])
            adj_crpt_fns.append(lambda A, p:gnu.dropout_edge(edge_index=A, p=p, force_undirected=True))

        self.feat_crpt = feat_crpt_fns
        self.adj_crpt = adj_crpt_fns

    def corrupt(self, type, dataset, p=.5):
        if type == 'feature':
            return [self.feat_crpt(data['x']) for data in dataset]
        elif type == 'adjacency':
            return [self.adj_crpt(data['edge_index'], p) for data in dataset]


class DGI(nn.Module):
    def __init__(self, xdim, adim, args):
        super(DGI, self).__init__()

        self.E_dim = args.E_dim

        if args.transductive == True:
            self.GCN = gnn.Sequential('x, edge_index',[
                (gnn.GCNConv(-1, self.E_dim), 'x, edge_index'),
                nn.ReLU(inplace=True)
            ]
            )
        else:
            raise NotImplementedError

        self.Readout = lambda input : F.sigmoid(input.sum(dim=0))
        self.W = nn.Linear(self.E_dim, self.E_dim)

    def encode(self, X, A):
        return self.GCN(X, A)
    
    def discriminate(self, h, s):
        return F.sigmoid(torch.dot(h, self.W(s)))
    
    # Check if the vectors change after one iteration