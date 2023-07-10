# Uses code from PMF

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

    path = os.path.join(args.basedir, args.expname, 'pmf')
    os.makedirs(path, exist_ok=True)  
    writer = SummaryWriter(path)

    dataset = movie_lens(args.datadir)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, drop_last=False)
    
    V = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
    Y = torch.rand(size=(args.D, args.N), device=device, requires_grad=True)

    optimizer = torch.optim.SGD([V, Y], lr=5e-3, momentum=.9)
    lu = args.lu
    lv = args.lv

    for i in trange(1, args.N_epoch+1):
        for batch in dataloader:
            optimizer.zero_grad() 

            users = batch['user'].to(device)
            movies = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            Y_= Y[:, users] 
            U = Y_
            U = torch.transpose(U, 0, 1) # U : (batch_size) * D
            V_ = V[:, movies]            # V_ : D * (batch_size)
            
            UTV = torch.sigmoid(torch.diagonal(torch.mm(U, V_))).to(device)
            
            loss = ratings_loss(UTV, ratings) + 0.5 * lu * (torch.norm(Y_, p='fro', dim=0)**2).sum() \
                                              + 0.5 * lv * (torch.norm(V_, p='fro', dim=0)**2).sum()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        
        RMSE = ratings_loss(UTV, ratings).sqrt()
        writer.add_scalar("RMSE/train", RMSE, i)
        if i%5 == 0:
            tqdm.write(f"[TRAIN] Iter: {i}, RMSE: {RMSE}")
    
    writer.flush()
    writer.close()

    torch.save({
        'V': V,
        'Y': Y,
        'lu': lu,
        'lv': lv
    }, os.path.join(path, f'{args.N_epoch}.pt'))

if __name__ == '__main__':
    train()
    