import sys,os
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from model import WRMF
from config import config_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    occf = WRMF(path=args.datadir, args=args)
    
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    
    occf.opt_X()
    occf.opt_Y()

    RMSE = occf.calc_pref()

    torch.save({
                'X': occf.X,
                'Y': occf.Y,
    }, os.path.join(path, f'{args.N_epoch}.pt'))

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    train(args)