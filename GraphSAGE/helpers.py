import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class GraphSAGE(nn.Module):
    '''
    path : metapath
    l : walk length
    d : vector dimension
    k : neighborhood size
    '''
    def __init__(self, prob, nbd, width, args):
        super(GraphSAGE, self).__init__()

        self.depth = args.depth
        self.prob = prob
        self.nbd = nbd

        self.aggregator = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.linears = nn.ModuleList(
            [nn.Linear(width) * args.depth]
        )

    def forward(self, x):
        '''
        x : (batch_size) * [node_idx, features.t()]
        '''
        def _create_bn(nbd, k, nodes): #Create B^K
            Bk = [set() for _ in range(k+1)]

            Bk[-1] = set(nodes.tolist())
            for i in range(k-1, -1, -1):
                s = set()
                for n in Bk[i+1]:
                    s = s.union(set(nbd[n]))
                Bk[i] = Bk[i+1].union(s)

            return Bk
        
        Bk = _create_bn(self.nbd, self.depth, x[:,0])
        Hk = dict.fromkeys(Bk[0])
        for k in Bk[0]:
            Hk[k] = torch.zeros(size=[self.depth+1, x[:,1:].size(-1)], device=x.device, dtype=x.dtype)
        for row in x:
            Hk[row[0]][0] = row[1:]

        for k in range(1, self.depth+1):
            for u in Bk[k]:
                nbd_list = []
                for nu in self.nbd[u]:
                    nbd_list.append(self.aggregator(Hk[nu][k-1]))
                agg_u, _ = torch.max(torch.stack(nbd_list), dim=0)
                Hk[u][k] = F.relu(self.linears[k-1](torch.cat([Hk[u][k-1], agg_u])))
                Hk[u][k] = F.normalize(Hk[u][k], dim=0)

        ret = []
        for row in x:
            ret.append(Hk[row[0]][-1])

        return torch.stack(ret)

    '''
    |
    | Need to modify
    V 
    '''
    def walk(self, starting_points, AWP, PPV, VPP, PWA):
        def _random_select(tf):
            ret = torch.zeros(tf.size(0), 1, dtype=int)

            for i in range(tf.size(0)):
                idx = tf[i].nonzero().squeeze(1)
                x = random.randint(0, idx.size(0))
                ret[i] = x

            return ret
        
        path = starting_points.clone().detach()

        for _ in range(0, self.l, 4): #Only considered APVPA
            p = AWP[1, _random_select(AWP[0, :] == path[:, -1].unsqueeze(1))]
            v = PPV[1, _random_select(PPV[0, :] == p)]
            p = VPP[1, _random_select(VPP[0, :] == v)]
            a = PWA[1, _random_select(PWA[0, :] == p)]

            path = torch.cat([path, a], dim=-1)

        return path

    def skipgram(self, path):
        def _negative_sample(prob, bound):
            samples = torch.tensor([], dtype=int)
            
            for i in range(bound.size(0)):
                idx = prob.multinomial(num_samples=self.k).unsqueeze(0)

                while torch.isin(idx, bound[i]).sum().item() != 0:
                    idx = prob.multinomial(num_samples=self.k).unsqueeze(0)
                samples = torch.cat([samples, idx], dim=0)
            
            return samples #[batch_size, N_neg]

        optimizer = optim.SGD([self.embedding.weight], lr=self.lr)
        
        for i in range(path.size(1)):
            lbd = max(0, i-self.k)
            rbd = min(path.size(1), i+self.k)
    
            for j in range(lbd, rbd):
                optimizer.zero_grad()
                pos = torch.log(
                        F.sigmoid(
                            (
                            self.embedding.weight[path[:,i]] *
                            self.embedding.weight[path[:,j]]
                            ).sum(dim=1).unsqueeze(1)
                        )
                    )
                
                neg_samples = _negative_sample(self.prob, path[:, lbd:rbd])
                neg = torch.sum(
                        torch.log(
                            F.sigmoid(
                            (
                            (-self.embedding.weight[neg_samples].transpose(0,1)) *
                            self.embedding.weight[path[:,i]]
                            ).sum(dim=2).unsqueeze(2)
                            )
                        )
                        ,dim=0)
                
                O_x = -(pos.mean() + neg.mean())
                O_x.backward()
                optimizer.step()
                
                          

        
