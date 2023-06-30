import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--basedir', type=str, help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, help='input data directory')

    parser.add_argument('--split', type=str, help='split type')
    parser.add_argument('--batch_sizee', type=int, default=256, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle batch')
    parser.add_argument('--N_epoch', type=int, default=100, help='number of epochs')

    parser.add_argument('--D', type=int, default=30, help='split dimension')
    parser.add_argument('--N', type=int, help='number of users')
    parser.add_argument('--M', type=int, help='number of movies')

    parser.add_argument('--constrained', type=bool, default=False, help='constrained or not')
    parser.add_argument('--adaptive', type=bool, default=False, help='adaptive or not')
    parser.add_argument('--lu', type=float, default=0.01, help='lambda_u')
    parser.add_argument('--lv', type=float, default=0.1, help='lambda_v')


    return parser