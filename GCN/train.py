import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL

from config import config_parser
from model import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()

def preprocessing(data):
    A = torch.eye(n=data.num_nodes, device=device)

    n1, n2 = data.edge_index[0], data.edge_index[1]
    A[n1, n2] = 1 # Self-loop added adjacency matrix
    
    D = torch.diag(torch.sum(A, dim=1).pow(-.5))
    A_ = torch.mm(torch.mm(D, A), D)

    return A_

def train():
    parser = config_parser()
    args = parser.parse_args()

    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    writer = SummaryWriter(path)

    if args.dataset == 'Planetoid':
        dataset = Planetoid(root=args.datadir, name=args.expname)
    elif args.dataset == 'NELL':
        dataset = NELL(root=args.datadir)

    data = dataset[0].to(device)
    A_ = preprocessing(data)
    Y = torch.zeros(size=(data.num_nodes, dataset.num_classes), device=device)
    idx = torch.arange(Y.shape[0], device=device)
    Y[idx, data.y] = 1

    model = GCN(A_, data.num_features, dataset.num_classes, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)

    model.train()
    for i in trange(1, args.N_epoch+1):
        optimizer.zero_grad()

        Z = F.softmax(model(data.x), dim=1)
        loss = loss_fn(Z[data.train_mask], Y[data.train_mask])
        
        correct = (Z.argmax(1) == data.y)
        Train_Accuracy = (correct[data.train_mask].sum()) / data.train_mask.sum()
        Val_Accuracy = (correct[data.val_mask].sum()) / data.val_mask.sum()

        writer.add_scalar('Train Accuracy', Train_Accuracy, i)
        writer.add_scalar('Val Accuracy', Val_Accuracy, i)
        writer.add_scalar('Training Loss', loss, i)

        if i%10 == 0:
            tqdm.write(f'[TRAIN] Iter: {i}, Accuracy: {Train_Accuracy.item():.2%}')

        loss.backward()
        optimizer.step()

    writer.flush()
    writer.close()

    torch.save({
                'global_step': args.N_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'{args.N_epoch}.pt'))

if __name__ == '__main__':
    train()