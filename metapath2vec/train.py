import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch_geometric.datasets import AMiner

from config import config_parser
from helpers import metapath2vec
from helpers import calc_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train(args):
    path = os.path.join(args.basedir, args.expname)
    os.makedirs(path, exist_ok=True)  
    
    if args.dataset == 'AMiner':
        dataset = AMiner(root=args.datadir)
    else:
        raise NotImplementedError
    
    data = dataset[0].to(device)
    data['author']['y'] = torch.where(data['author']['y'] < 100000, data['author']['y'], 0)
    data['author']['y_index'] = data['author']['y_index'][:, torch.any(data['author']['y'] != 0, dim=0)].squeeze()
    data['author']['y'] = data['author']['y'][:, torch.any(data['author']['y'] != 0, dim=0)].squeeze()
    num_authors = data['author']['num_nodes'] = 100000

    data['paper']['num_nodes'] = 200000

    data[('paper', 'written_by', 'author')]['edge_index'] = torch.where(data[('paper', 'written_by', 'author')]['edge_index'][0] < 200000, 
                                                                    data[('paper', 'written_by', 'author')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('paper', 'written_by', 'author')]['edge_index'] = torch.where(data[('paper', 'written_by', 'author')]['edge_index'][1] < 100000, 
                                                                    data[('paper', 'written_by', 'author')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('paper', 'written_by', 'author')]['edge_index'] = data[('paper', 'written_by', 'author')]['edge_index'][:, torch.any(data[('paper', 'written_by', 'author')]['edge_index'] != 0, dim=0)]
    
    data[('author', 'writes', 'paper')]['edge_index'] = torch.where(data[('author', 'writes', 'paper')]['edge_index'][0] < 100000, 
                                                                    data[('author', 'writes', 'paper')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('author', 'writes', 'paper')]['edge_index'] = torch.where(data[('author', 'writes', 'paper')]['edge_index'][1] < 200000, 
                                                                    data[('author', 'writes', 'paper')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('author', 'writes', 'paper')]['edge_index'] = data[('author', 'writes', 'paper')]['edge_index'][:, torch.any(data[('author', 'writes', 'paper')]['edge_index'] != 0, dim=0)]
    
    data[('paper', 'published_in', 'venue')]['edge_index'] = torch.where(data[('paper', 'published_in', 'venue')]['edge_index'][0] < 200000, 
                                                                    data[('paper', 'published_in', 'venue')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('paper', 'published_in', 'venue')]['edge_index'] = data[('paper', 'published_in', 'venue')]['edge_index'][:, torch.any(data[('paper', 'published_in', 'venue')]['edge_index'] != 0, dim=0)]

    data[('venue', 'publishes', 'paper')]['edge_index'] = torch.where(data[('venue', 'publishes', 'paper')]['edge_index'][1] < 200000, 
                                                                    data[('venue', 'publishes', 'paper')]['edge_index'],
                                                                    torch.zeros(2,1, dtype=int, device=device))
    data[('venue', 'publishes', 'paper')]['edge_index'] = data[('venue', 'publishes', 'paper')]['edge_index'][:, torch.any(data[('venue', 'publishes', 'paper')]['edge_index'] != 0, dim=0)]

    print('dataset cropped')
   
    #y : gt category
    #y_idx : idx of the node
    
    #prob = torch.ones((3,3))
    prob = calc_prob(
        data[('author', 'writes', 'paper')]['edge_index'],
        num_authors,
        device
    )
    
    print('Probability distribution complete')

    if args.is_meta:
        model = metapath2vec(
            data['author']['num_nodes'],
            data['venue']['num_nodes'], 
            data['paper']['num_nodes'],
            prob,
            args)
    else:
        raise NotImplementedError
    
    model = model.to(device)
    batch_size_ = args.batch_size
    #writer = SummaryWriter(path)
    
    for e in trange(1, args.N_walk+1):
        #losses = []
        for i in range(0, num_authors, batch_size_):
            batch_size = batch_size_ if i+batch_size_ < num_authors else num_authors-i
            path = model.walk(
                torch.arange(i, i+batch_size).unsqueeze(1).to(device),
                data[('author', 'writes', 'paper')]['edge_index'],
                data[('paper', 'published_in', 'venue')]['edge_index'],
                data[('venue', 'publishes', 'paper')]['edge_index'],
                data[('paper', 'written_by', 'author')]['edge_index']
            ).to(device)
            model.skipgram(path)
            

        #writer.add_scalar('Train loss', sum(losses)/len(losses), e)
        #if e%10 == 0:
            #tqdm.write(f'[TRAIN] Epoch: {e}, Loss: {sum(losses)/len(losses):.4f}')

    #writer.flush()
    #writer.close()

    torch.save(model.state_dict(), os.path.join(path, f'{args.N_walk}.pt'))

if __name__ == '__main__':
    torch.set_default_device(device)
    parser = config_parser()
    args = parser.parse_args()

    train(args)