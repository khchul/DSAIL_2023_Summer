import sys,os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import ml_data

from ml_data import movie_lens
from config import config_parser

ratings_loss = lambda x, y: .5 * torch.sum((x - y)**2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    parser = config_parser()
    args = parser.parse_args()  
    writer = SummaryWriter(args.basedir)

    dataset = movie_lens(args.datadir, args.split)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, drop_last=False)
    
    V = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
    Y = torch.rand(size=(args.D, args.N), device=device, requires_grad=True)
    if args.constrained == False:
        W = torch.zeros_like(V, device=device)
    else:
        W = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([V, Y, W], lr=1e-3)

    for i in trange(args.N_epoch):
        for batch in dataloader:
            optimizer.zero_grad() 

            users = batch['user'].to(device)
            movies = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            U = Y[:, users] + W[:, movies]
            U = torch.transpose(U, 0, 1) # U : (batch_size) * D
            V_ = V[:, movies]            # V_ : D * (batch_size)
            
            UTV = torch.diagonal(torch.mm(U, V_)).to(device)
            loss = ratings_loss(UTV, ratings) + args.lu * torch.norm(U, p='fro', dim=1).sum() \
                                             + args.lv * torch.norm(V_, p='fro', dim=0).sum()

            loss.backward()
            optimizer.step()
        
        RMSE = ratings_loss(UTV, ratings).sqrt()
        writer.add_scalar("RMSE/train", RMSE, i)
        if i%5 == 0:
            tqdm.write(f"[TRAIN] Iter: {i} RMSE: {RMSE}")
    
    writer.flush()
    writer.close()

if __name__ == '__main__':
    #torch.set_default_device(device)
    train()
    