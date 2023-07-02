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

    if args.constrained:
        path = os.path.join(args.basedir, args.expname, 'cpmf')
    elif args.adaptive:
        path = os.path.join(args.basedir, args.expname, 'pmfa')
    else:
        path = os.path.join(args.basedir, args.expname, 'pmf')
    os.makedirs(path, exist_ok=True)  
    writer = SummaryWriter(path)

    dataset = movie_lens(args.datadir)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=args.shuffle, drop_last=False)
    
    V = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)
    Y = torch.rand(size=(args.D, args.N), device=device, requires_grad=True)
    if args.constrained == False:
        W = torch.zeros_like(V, device=device)
    else:
        W = torch.rand(size=(args.D, args.M), device=device, requires_grad=True)

    optimizer = torch.optim.SGD([V, Y, W], lr=5e-3, momentum=.9)
    lu = torch.tensor([args.lu], device=device, requires_grad=True)
    lv = torch.tensor([args.lv], device=device, requires_grad=True)
    lw = torch.tensor([args.lw], device=device)

    for i in trange(1, args.N_epoch+1):
        for batch in dataloader:
            optimizer.zero_grad() 

            users = batch['user'].to(device)
            movies = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            Y_= Y[:, users] 
            W_ = W[:, movies]
            U = Y_ + W_
            U = torch.transpose(U, 0, 1) # U : (batch_size) * D
            V_ = V[:, movies]            # V_ : D * (batch_size)
            
            UTV = torch.sigmoid(torch.diagonal(torch.mm(U, V_))).to(device)
            if args.adaptive and i%args.N_a == 0:
                lu.requires_grad = True
                lv.requires_grad = True
                V.requires_grad = False
                Y.requires_grad = False
                if args.constrained:
                    W.requires_grad = False
            else:
                lu.requires_grad = False
                lv.requires_grad = False
                V.requires_grad = True
                Y.requires_grad = True
                if args.constrained:
                    W.requires_grad = True

            loss = ratings_loss(UTV, ratings) + 0.5 * lu * (torch.norm(Y_, p='fro', dim=0)**2).sum() \
                                              + 0.5 * lv * (torch.norm(V_, p='fro', dim=0)**2).sum() \
                                              + 0.5 * lw * (torch.norm(W_, p='fro', dim=0)**2).sum()
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
        'W': W,
        'lu': lu,
        'lv': lv,
        'lw': lw
    }, os.path.join(path, f'{args.N_epoch}.pt'))

if __name__ == '__main__':
    train()
    