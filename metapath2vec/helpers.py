import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def calc_prob(AWP, num_author, device):
    prob = torch.zeros(num_author, device=device)
    
    for i in range(AWP.size(1)):
        prob[AWP[0][i]] += 1
    
    prob = torch.pow(prob, .75) / math.pow(AWP.size(1), .75)

    return prob

class metapath2vec(nn.Module):
    '''
    path : metapath
    l : walk length
    d : vector dimension
    k : neighborhood size
    '''
    def __init__(self, N_author, N_venue, N_paper, prob, args):
        super(metapath2vec, self).__init__()

        self.N_author = N_author
        self.N_venue = N_venue
        self.N_paper = N_paper
        self.N_total = N_author + N_venue + N_paper
        self.prob = prob

        self.path = args.metapath
        self.l = args.walk_len
        self.d = args.d
        self.k = args.neighborhood
        self.lr = args.lr
        
        # Dim : [author, venue, paper] x d
        self.embedding = nn.Embedding(num_embeddings=self.N_total, embedding_dim=self.d)

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
                
                          

        
