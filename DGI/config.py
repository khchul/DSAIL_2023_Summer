import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--datadir', type=str, help='data directory')
    parser.add_argument('--testname', type=str, help='test name')
    parser.add_argument('--split', type=str, help='dataset split type')

    parser.add_argument('--L_type', type=str, help='Learning type')
    parser.add_argument('--C_type', type=str, default='feature', help='Corruption type')

    parser.add_argument('--batch_size', type=int, default=256, help='Number of nodes in a batch')
    parser.add_argument('--N_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0, help='L2 regularizer')
    parser.add_argument('--E_dim', type=int, default=256, help='Dimension of vector embeddings')
    parser.add_argument('--p', type=float, default=.5, help='Corruption rate for adjacency corruption')

    return parser