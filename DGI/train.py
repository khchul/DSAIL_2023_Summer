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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transductive_learning(dataset, path):
    

    return

def inductive_learning(dataset, path):

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

    if args.transductive == True:
        transductive_learning(dataset, path)
    else:
        inductive_learning(dataset, path)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    train(args) 