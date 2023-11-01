import sys,os

from config import config_parser

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader
from helpers import DGI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def visualize(args):
    model_path = os.path.join(args.basedir, args.expname, args.testname)
    summary_path = os.path.join(args.basedir, args.expname)
    writer = SummaryWriter(summary_path)
    
    F_dim, dataloader = get_dataloader(args, split=args.split)
    
    model = DGI(F_dim, args.E_dim, args.L_type).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    for batch in dataloader:
        if args.L_type == 'transductive':
            batch_size = batch['x'].size(0)
        else:
            batch_size = args.batch_size

        batch = batch.to(device)
        embeddings = model.encode(batch['x'], batch['edge_index'])[:batch_size]

        writer.add_embedding(
            embeddings,
            batch['y'][:batch_size]
        )
    
    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    visualize(args)
