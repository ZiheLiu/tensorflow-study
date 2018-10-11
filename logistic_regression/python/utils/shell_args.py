import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--optimizer', type=str, default='newton')
    parser.add_argument('--feature_0', type=int, default=0)
    parser.add_argument('--feature_1', type=int, default=1)


_parser = argparse.ArgumentParser()
add_arguments(_parser)
SHELL_ARGS = _parser.parse_args()
