import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--datadir', type=str, help='data directory')

    parser.add_argument('--depth', type=int, default=2, help='Size of depth')
    parser.add_argument('--N_walk', type=int, default=150, help='Number of walks per node')
    parser.add_argument('--walk_len', type=int, default= 12, help='Walk length')
    parser.add_argument('--neighborhood', type=int, default=3, help='Neighborhood size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--d', type=int, default=10, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--N_negative', type=int, default=3, help='Size of negative samples')

    #Test args
    parser.add_argument('--testname', type=str, help='Name of the trained tensors')
    parser.add_argument('--hit_size', type=int, default=10, help='Neighborhood size')
    parser.add_argument('--chunk', type=int, default=64, help='Batch chunk size')

    return parser