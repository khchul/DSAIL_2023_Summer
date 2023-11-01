import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from config import config_parser

from data import get_dataloader
from helpers import Corrupter
from helpers import DGI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    F_dim, dataloader = get_dataloader(args, split=args.split)

    model = DGI(F_dim, args.E_dim, args.L_type).to(device)
    corrupter = Corrupter()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    writer = SummaryWriter(path)

    for i in trange(1, args.N_epoch+1):
        losses = []
        for batch in dataloader: # Only one pos & neg sample
            batch = batch.to(device)
            crpt_batch = corrupter.corrupt(args.C_type, batch, args.p)
 
            optimizer.zero_grad()

            if args.L_type == 'transductive':
                batch_size = batch['x'].size(0)
            else:
                batch_size = args.batch_size
            
            pos_H = model.encode(batch['x'], batch['edge_index'])[:batch_size]
            summary = model.readout(pos_H)
            neg_H = model.encode(crpt_batch['x'], crpt_batch['edge_index'])[:batch_size]

            pos_D = model.discriminate(pos_H, summary)
            neg_D = model.discriminate(neg_H, summary)
            target = torch.cat([torch.ones_like(pos_D), torch.zeros_like(neg_D)])
            
            loss = F.binary_cross_entropy_with_logits(torch.cat([pos_D, neg_D]), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        writer.add_scalar('Train loss', sum(losses)/len(losses), i)
        if i%10 == 0:
            tqdm.write(f'[TRAIN] Epoch: {i}: {loss}')

    writer.flush()
    writer.close()

    torch.save({
                'model_state_dict': model.state_dict()
    }, os.path.join(path, f'{args.L_type}_{args.C_type}_{args.N_epoch}.pt'))

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
 
    train(args) 