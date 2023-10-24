import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI

from config import config_parser

from helpers import Corrupter
from helpers import DGI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transductive_learning(train_data, path, args):

    return

def inductive_learning(train_data, path, args):

    return


def train(args):
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    
    if args.dataset == 'Planetoid':
        dataset = Planetoid(root=args.datadir, name=args.expname)
    elif args.dataset == 'PPI':
        dataset = PPI(root=args.datadir)
    else:
        raise NotImplementedError

    crpt = Corrupter(len(dataset))
    if args.feature == True:
        crpt_X_list = crpt.feat_crpt(dataset)
    if args.adjacency == True:
        crpt_A_list = crpt.adj_crpt(dataset)

    train_data = {
        'X_list': [data['x'] for data in dataset],
        'A_list': [data['edge_index'] for data in dataset],
        'crpt_X_list': crpt_X_list if args.feature else None,
        'crpt_A_list': crpt_A_list if args.adjacency else None
    }

    if args.transductive == True:
        transductive_learning(train_data, path, args)
    else:
        inductive_learning(train_data, path, args)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
 
    train(args) 