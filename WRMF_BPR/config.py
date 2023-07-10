import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, help='input data directory')

    parser.add_argument('--N_epoch', type=int, default=100, help='number of epochs')

    parser.add_argument('--N_user', type=int, help='number of users')
    parser.add_argument('--N_movie', type=int, help='number of movies')
    parser.add_argument('--f', type=int, default=100, help='number of features')
    
    parser.add_argument('--alpha', type=int, help='alpha')
    parser.add_argument('--eps', type=float, default=1e-6, help='epsilon')
    parser.add_argument('--l', type=float, default=0.001, help='lambda')

    return parser