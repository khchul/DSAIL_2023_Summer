import sys,os
import matplotlib.pyplot as plt

from config import config_parser

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch_geometric.datasets import WordNet18RR
from torch_geometric.datasets import FB15k_237

from config import config_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hits = 0

@torch.no_grad()

def batchwise_test(TransE, test_set, k, p, chunk=64):
    global hits

    num_nodes = test_set['num_nodes']
    num_relations = test_set['num_relations']

    for i in range(0, num_relations, chunk):
        batch_size = chunk if i+chunk < num_relations else num_relations-i

        #check for heads
        corrupt_head = torch.arange(0, num_nodes).repeat(2, batch_size, 1).to(device)
        corrupt_head[1] = test_set['eset'][:,i:i+batch_size][1].unsqueeze(1).repeat(1, num_nodes).to(device)
        dist = TransE['E'][corrupt_head[0]] + TransE['L'][test_set['lset'][i:i+batch_size]][:,None,:] - TransE['E'][corrupt_head[1]]
        dist = dist.norm(p=p, dim=-1)
        
        _, knn = dist.sort()
        knn = knn.cpu()
        hits += torch.sum(torch.isin(test_set['eset'][:,i:i+batch_size][0], knn[:,:k]))

        #check for tails
        corrupt_head = torch.arange(0, num_nodes).repeat(2, batch_size, 1).to(device)
        corrupt_head[0] = test_set['eset'][:,i:i+batch_size][0].unsqueeze(1).repeat(1, num_nodes).to(device)
        dist = TransE['E'][corrupt_head[0]] + TransE['L'][test_set['lset'][i:i+batch_size]][:,None,:] - TransE['E'][corrupt_head[1]]
        dist = dist.norm(p=p, dim=-1)
        _, knn = dist.sort()
        
        knn = knn.cpu()
        hits += torch.sum(torch.isin(test_set['eset'][:,i:i+batch_size][1], knn[:,:k]))
        


def test(args):
    path = os.path.join(args.basedir, args.expname, args.testname)
    TransE = torch.load(path)
    if args.dataset == 'WordNet':
        dataset = WordNet18RR(root=args.datadir)

    elif args.dataset == 'Freebase':
        dataset = FB15k_237(root=args.datadir)

    print('Length of each relation tensors')
    print(TransE['L'].norm(dim=1))

    test_set = {}
    test_set['eset'] = dataset[0]['edge_index'][:, dataset[0]['test_mask']]
    test_set['lset'] = dataset[0]['edge_type'][dataset[0]['test_mask']]
    l = test_set['num_relations'] = dataset[0]['test_mask'].sum().item()
    test_set['num_nodes'] = dataset[0]['num_nodes']

    batchwise_test(TransE, test_set, args.hit_size, args.norm, args.chunk)

    print(f'Hit@{args.hit_size}: {(hits / l):.2%}')
    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    test(args)