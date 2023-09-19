import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL

from config import config_parser
from helpers import GraphSAGE
from helpers import calc_prob
from helpers import get_nbd

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
        'x':data['x'],
        'y':data['y'][data['train_mask']],
        'edge_index':data['edge_index'],
        'Num_nodes':data['train_mask'].shape[0]
    }.to(device)
    
    prob = calc_prob(
        train_data['Num_nodes'],
        train_data['edge_index'],
        device
    )
    print('Probability distribution complete')

    nbd = get_nbd(train_data['edge_index'])
    model = GraphSAGE(prob, nbd, train_data['x'].size(1), args)
    model = model.to(device)

    #######
    batch_size_ = args.batch_size

    for e in trange(1, args.N_walk+1):
        for i in range(0, num_authors, batch_size_):
            batch_size = batch_size_ if i+batch_size_ < num_authors else num_authors-i
            walk_path = model.walk(
                torch.arange(i, i+batch_size).unsqueeze(1).to(device),
                data[('author', 'writes', 'paper')]['edge_index'],
                data[('paper', 'published_in', 'venue')]['edge_index'],
                data[('venue', 'publishes', 'paper')]['edge_index'],
                data[('paper', 'written_by', 'author')]['edge_index']
            ).to(device)
            model.skipgram(walk_path)

    torch.save(model.state_dict(), os.path.join(path, f'{args.N_walk}.pt'))

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    train(args) 