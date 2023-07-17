import os, sys
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
triplet_loss = lambda margin, norm : nn.TripletMarginLoss(margin=margin, p=norm, reduction='sum')

def init_tensors(num_nodes, num_edge_type, dim):
    l = F.normalize((6. / math.sqrt(dim) * (2. * torch.rand(num_edge_type, dim) - 1))).to(device)
    e = 6. / math.sqrt(dim) * (2. * torch.rand(num_nodes, dim) - 1).to(device)

    e.requires_grad_(True)
    l.requires_grad_(True)

    return e, l

def train(args):
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    
    if args.dataset == 'WordNet':
        dataset = WordNet18RR(root=args.datadir)

    elif args.dataset == 'Freebase':
        dataset = FB15k_237(root=args.datadir)

    train_eset = dataset[0]['edge_index'][:, dataset[0]['train_mask']]
    train_lset = dataset[0]['edge_type'][dataset[0]['train_mask']]
    num_relation_types = dataset[0]['edge_type'].max().item() + 1
    num_relations = dataset[0]['train_mask'].sum().item()
    num_nodes = dataset[0]['num_nodes']
    E, L = init_tensors(num_nodes, num_relation_types , args.k)
    
    batch_size_ = args.batch_size
    optimizer = torch.optim.SGD([E, L], lr=args.lr)
    loss_fn = triplet_loss(args.r, args.norm)
    writer = SummaryWriter(path)

    for e in trange(1, args.N_epoch+1):
        losses = []
        for i in range(0, num_relations, batch_size_):
            batch_size = batch_size_ if i+batch_size_ < num_relations else num_relations-i

            h = E[train_eset[0, i:i+batch_size]]
            t = E[train_eset[1, i:i+batch_size]]
            l = L[train_lset[i:i+batch_size]]
            
            corrupted_idx = (torch.rand(batch_size) * (num_nodes-1) + 1).to(int)
            if torch.rand(1) < .5:
                h_ = E[(train_eset[0, i:i+batch_size] + corrupted_idx) % num_nodes]
                t_ = t
            else:
                h_ = h
                t_ = E[(train_eset[1, i:i+batch_size] + corrupted_idx) % num_nodes]
            
            E.requires_grad_(False)
            E[train_eset[:, i:i+batch_size]] = F.normalize(E[train_eset[:, i:i+batch_size]], dim=-1)
            E[(train_eset[:, i:i+batch_size]+corrupted_idx) % num_nodes] \
                = F.normalize(E[(train_eset[:, i:i+batch_size]+corrupted_idx) % num_nodes], dim=-1)
            E.requires_grad_(True)

            if args.l_norm:
                L.requires_grad_(False)
                L[train_lset[i:i+batch_size]] = F.normalize(L[train_lset[i:i+batch_size]])
                L.requires_grad_(True)

            optimizer.zero_grad()
            loss = loss_fn(l, t-h, t_ - h_)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        writer.add_scalar('Train loss', sum(losses)/len(losses), e)
        if e%10 == 0:
            tqdm.write(f'[TRAIN] Epoch: {e}, Loss: {sum(losses)/len(losses):.4f}')

    writer.flush()
    writer.close()

    torch.save({
                'Dataset': args.dataset,
                'Number_of_nodes':num_nodes,
                'Number_of_edge_types':num_relation_types,
                'E': E,
                'L': L,
                'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'{args.N_epoch}.pt'))

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    train(args)