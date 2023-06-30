import sys,os
import torch
from torch.utils.data import Dataloader
from tqdm import tqdm, trange

import ml_data
import config

from .ml_data import movie_lens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    parser = config_parser()
    args = parser.parse_args()  

    dataset = movie_lens(args.datadir, args.split)
    dataloader = Dataloader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, drop_last=False)
    
    if args.constrained == False:
        U = torch.rand(size=(args.D, args.N), device=device, requires_grad=True)
        V = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
        
        for i in trange(args.N_epoch):
            for batch in tqdm(dataloader):
                users = batch['user']
                movies = batch['movie']
                ratings = batch['rating']
    else:
        Y = torch.rand(size=(args.D, args.N), device=device, requires_grad=True)
        W = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
        V = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()