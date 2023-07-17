import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--datadir', type=str, help='data directory')

    parser.add_argument('--norm', type=int, default='2', help='Degree of norm')
    parser.add_argument('--N_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--k', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--r', type=float, default=1., help='Minimum margin')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    
    parser.add_argument('--l_norm', action='store_true', help='Normalize relation')

    #Test args
    parser.add_argument('--testname', type=str, help='Name of the trained tensors')
    parser.add_argument('--hit_size', type=int, default=10, help='Neighborhood size')
    parser.add_argument('--chunk', type=int, default=64, help='Batch chunk size')

    return parser