import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--prefix', type=str, default='static/input/iris')
    parser.add_argument('--raw_data_path', type=str, default='static/input/iris')


_parser = argparse.ArgumentParser()
add_arguments(_parser)
SHELL_ARGS = _parser.parse_args()
