from torch_geometric.datasets import PPI, Planetoid
from torch_geometric.loader import DataLoader, NeighborLoader, DenseDataLoader
from torch_geometric.data import Batch

def get_dataloader(args, split):
    if args.dataset == 'Planetoid':
        dataset = Planetoid(root=args.datadir, name=args.expname, split=split)
    elif args.dataset == 'PPI':
        dataset = PPI(root=args.datadir, split=split)
    else:
        raise NotImplementedError

    if len(dataset) > 1:
        data = Batch.from_data_list(dataset)
    else:
        data = dataset[0]
    
    F_dim = data['x'].size(1)
    if args.L_type == 'transductive':
        dataloader = DataLoader(dataset, shuffle=True)
    else:
        dataloader = NeighborLoader(data, num_neighbors=[10, 75, 500], batch_size=args.batch_size, shuffle=True)

    return F_dim, dataloader