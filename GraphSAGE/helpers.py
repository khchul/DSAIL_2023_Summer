import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_nbd(edges):
    nbd = {}

    for e in edges.t():
        if e[0].item() in nbd:
            nbd[e[0].item()].append(e[1].item())
        else:
            nbd[e[0].item()] = [e[1].item()]

        if e[1].item() in nbd:
            nbd[e[1].item()].append(e[0].item())
        else:
            nbd[e[1].item()] = [e[0].item()]

    return nbd

def calc_prob(num_nodes, edges, device):
    '''
    Calculate probabilities
    for negative sampling
    '''
    prob = torch.zeros(num_nodes, dtype=int, device=device)
    nodes, counts = torch.unique(edges, return_counts=True)
    prob[nodes] = counts
    
    prob = torch.pow(prob, .75) / math.pow(edges.size(1)*2, .75)

    return prob

def walk(walk_len, nbd, starting_points):
    path = starting_points.clone().detach()

    for _ in range(walk_len):
        next_nodes = []
        for point in path[:,0].tolist():
            next_nodes.append(random.choice(nbd[point]))

        next_nodes = torch.tensor(next_nodes)[:,None].to(path.device)
        path = torch.cat([path, next_nodes], dim=-1)

    return path

def negative_sampling(path, prob, N_nodes, N_samples):
    samples = torch.tensor([], dtype=int).to(path.device)
        
    for i in range(path.size(0)):
        idx = prob.multinomial(num_samples=N_samples)[None,:].to(path.device)

        while torch.isin(idx, path[i]).sum().item() != 0:
            idx = prob.multinomial(num_samples=N_samples)[None,:].to(path.device)
        samples = torch.cat([samples, idx], dim=0)
    
    return samples #[batch_size, N_neg]

class GraphSAGE(nn.Module):
    def __init__(self, width, depth):
        super(GraphSAGE, self).__init__()

        self.depth = depth
        self.aggregator = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.linears = nn.ModuleList(
            [nn.Linear(width*2, width)] * depth
        )

    def forward(self, nbd, x):
        '''
        x : (batch_size) * [node_idx, features.t()]
        ret : (batch_size) * width
        '''
        torch.autograd.set_detect_anomaly(True)
        def _create_bn(k, nodes): #Create B^K
            Bk = [set() for _ in range(k+1)]

            Bk[-1] = set(nodes.tolist())
            for i in range(k-1, -1, -1):
                s = set()
                for n in Bk[i+1]:
                    s = s.union(set(nbd[n]))
                Bk[i] = Bk[i+1].union(s)

            return Bk
        
        Bk = _create_bn(self.depth, x[:,0])
        Hk = dict.fromkeys(Bk[0])
        for k in Bk[0]:
            Hk[k] = torch.zeros(size=[self.depth+1, x[:,1:].size(-1)], device=x.device, dtype=x.dtype, requires_grad=False)
        for row in x:
            Hk[int(row[0].item())][0] = row[1:]

        for k in range(1, self.depth+1):
            for u in Bk[k]:
                #nbd_list = []
                nbd_list = torch.tensor([]).to(x.device)
                for nu in nbd[u]:
                    nbd_list = torch.cat([nbd_list, self.aggregator(Hk[nu][k-1])[None,:]], dim=0)
                    #nbd_list.append(self.aggregator(Hk[nu][k-1]))   
                #agg_u, _ = torch.max(torch.stack(nbd_list), dim=0)
                agg_u, _ = torch.max(nbd_list, dim=0)

                Hk[u][k] = F.normalize(
                    F.relu(self.linears[k-1](torch.cat([Hk[u][k-1], agg_u]))), dim=0
                    )
            
        ret = []
        for row in x:
            ret.append(Hk[int(row[0].item())][-1])

        return torch.stack(ret)                      

        
