import sys,os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import ml_data

from ml_data import movie_lens
from config import config_parser

ratings_loss = lambda x, y: .5 * torch.sum((x - y)**2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()

def calculate_rmse(dict, data_path):
    Y = dict['Y']
    V = dict['V']
    W = dict['W']
    lu = dict['lu']
    lv = dict['lv']
    lw = dict['lw']

    dataset = movie_lens(data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    RMSE = 0
    for batch in dataloader:
        users = batch['user'].to(device)
        movies = batch['movie'].to(device)
        ratings = batch['rating'].to(device)

        Y_= Y[:, users] 
        W_ = W[:, movies]
        U = Y_ + W_
        U = torch.transpose(U, 0, 1) 
        V_ = V[:, movies]           
            
        UTV = torch.sigmoid(torch.diagonal(torch.mm(U, V_))).to(device)
        
        RMSE += ratings_loss(UTV, ratings).sqrt()
    
    return (RMSE / (len(dataloader))).cpu().numpy()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    pmf_path = os.path.join(args.basedir, args.expname, 'pmf', f'{args.N_epoch}.pt')
    pmfa_path = os.path.join(args.basedir, args.expname, 'pmfa', f'{args.N_epoch}.pt')
    cpmf_path = os.path.join(args.basedir, args.expname, 'cpmf', f'{args.N_epoch}.pt')

    pmf = torch.load(pmf_path)
    pmfa = torch.load(pmfa_path)
    cpmf = torch.load(cpmf_path)

    pmf_1_50 = calculate_rmse(pmf, '../datasets/ml-100k/u1_1-50.test')
    pmfa_1_50 = calculate_rmse(pmfa, '../datasets/ml-100k/u1_1-50.test')
    cpmf_1_50 = calculate_rmse(cpmf, '../datasets/ml-100k/u1_1-50.test')

    pmf_51 = calculate_rmse(pmf, '../datasets/ml-100k/u1_51-.test')
    pmfa_51 = calculate_rmse(pmfa, '../datasets/ml-100k/u1_51-.test')
    cpmf_51 = calculate_rmse(cpmf, '../datasets/ml-100k/u1_51-.test')

    plt.figure(figsize = (10, 5))
    plt.title('Movielens dataset')

    X1=[1,3]
    data1 = [pmf_1_50, pmf_51]
    plt.bar(X1, data1,color='r',width=0.5, label='pmf')

    X2=[1+0.5,3+0.5]
    data2 = [pmfa_1_50, pmfa_51]
    plt.bar(X2, data2,color='g',width=0.5, label='pmfa')

    X3=[1+1,3+1]
    data3 = [cpmf_1_50, cpmf_51]
    plt.bar(X3, data3,color='b',width=0.5, label='cpmf')

    ticklabel=['User 1-50', 'User 51-']
    plt.xticks(X2,ticklabel,fontsize=15,rotation=0)
    plt.tick_params(axis='x', bottom=False)
    plt.legend(ncol=2)

    plt.show()

