import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--datadir', type=str, help='data directory')

    parser.add_argument('--transductive', action='store_true', help='Transductive learning')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--N', type=int, default=1, help='Number of positive samples')
    parser.add_argument('--M', type=int, default=1, help='Number of negative samples')

    #Test args
    parser.add_argument('--testname', type=str, help='Name of the trained tensors')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch chunk size')

    return parser