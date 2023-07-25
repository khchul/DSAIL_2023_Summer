import sys,os

from config import config_parser

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import AMiner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(args):

    model_path = os.path.join(args.basedir, args.expname, args.testname)
    model = torch.load(model_path)
    
    if args.dataset == 'AMiner':
        dataset = AMiner(root=args.datadir)
    else:
        raise NotImplementedError

    test_embeddings = model['embedding.weight'][dataset[0]['author']['y_index']]
    test_labels = dataset[0]['author']['y']

    path = os.path.join(args.basedir, args.expname)
    writer = SummaryWriter(path)
    writer.add_embedding(
        test_embeddings,
        test_labels
    )
    
    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    test(args)
