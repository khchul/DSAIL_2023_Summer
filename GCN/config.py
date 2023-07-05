import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, help='data directory')

    parser.add_argument('--N_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--F', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--dropout', action='store_true', help='Perform dropout')
    parser.add_argument('--dropout_rate', type=float, default=.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--L2', type=float, default=1e-5, help='L2 Regularizaiton hparam')


    return parser