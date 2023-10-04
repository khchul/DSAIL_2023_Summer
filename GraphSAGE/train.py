import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL

from config import config_parser
from helpers import GraphSAGE
from helpers import calc_prob
from helpers import get_nbd
from helpers import walk
from helpers import negative_sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    
    if args.dataset == 'Planetoid':
        dataset = Planetoid(root=args.datadir, name=args.expname)
    elif args.dataset == 'NELL':
        dataset = NELL(root=args.datadir)
    else:
        raise NotImplementedError

    data = dataset[0].to(device)
    train_data = {
        'x':data['x'].to(device),
        'y':data['y'][data['train_mask']].to(device),
        'edge_index':data['edge_index'].to(device),
        'Num_nodes':data['train_mask'].shape[0]
    }
    num_nodes = train_data['Num_nodes']

    prob = calc_prob(
        num_nodes,
        train_data['edge_index'],
        device
    )
    print('Probability distribution complete')

    nbd = get_nbd(train_data['edge_index'])
    model = GraphSAGE(train_data['x'].size(1), args.depth)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.9)

    for i in trange(1, args.N_walk+1):
        print('Shuffle data before a random walk')
        indices = torch.randperm(num_nodes, device=device)

        for i_batch in range(0, num_nodes, args.batch_size):
            starting_points = indices[i_batch:i_batch+args.batch_size][:,None]
            U = torch.cat([starting_points, train_data['x'][indices[i_batch:i_batch+args.batch_size]]], dim=1)
            path = walk(args.walk_len, nbd, starting_points)
            neg_samples = negative_sampling(path, prob, num_nodes, args.N_negative)
            
            for v in range(1, args.walk_len):
                pos_node = path[:,v][:,None]
                V = torch.cat([pos_node, train_data['x'][path[:,v]]], dim=1)

                optimizer.zero_grad()
                Z_u = model(nbd, U)
                Z_v = model(nbd, V)

                pos = -torch.log(
                    F.sigmoid((Z_u * Z_v).sum(dim=1).unsqueeze(1))
                )

                neg = torch.zeros_like(pos, device=device)
                for s in range(args.N_negative):
                    neg_batch = neg_samples[:,s][:,None]
                    N = torch.cat([neg_batch, train_data['x'][neg_samples[:,s]]], dim=1)
                    Z_n = model(nbd, N)

                    neg = neg - torch.log(
                        F.sigmoid((Z_u * Z_n).sum(dim=1).unsqueeze(1))
                    )

                J = pos.mean() + neg.mean()
                J.backward()
                optimizer.step()
                
        tqdm.write(f'[TRAIN] Epoch: {i},  : {J:.4f}')

    torch.save(model.state_dict(), os.path.join(path, f'{args.N_walk}.pt'))

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    train(args) 