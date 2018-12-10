import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--prefix', type=str, default='iris')
    parser.add_argument('--initializer', type=str, default='kmeans++')
    parser.add_argument('--epochs', type=int, default=10)


_parser = argparse.ArgumentParser()
add_arguments(_parser)
SHELL_ARGS = _parser.parse_args()
