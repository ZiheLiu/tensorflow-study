import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--raw_data_path', type=str, default='static/input/iris')

    parser.add_argument('--prefix', type=str, default='static/input/iris')
    parser.add_argument('--hidden_sizes', type=str, default='32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.01)


_parser = argparse.ArgumentParser()
add_arguments(_parser)
SHELL_ARGS = _parser.parse_args()

SHELL_ARGS.hidden_sizes = [int(hidden_size) for hidden_size in SHELL_ARGS.hidden_sizes.split(',')]
