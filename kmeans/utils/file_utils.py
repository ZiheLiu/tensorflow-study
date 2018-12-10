import os

import constants


def safe_mkdirs(dir_name):
    """Create dir if the dir does not exist."""
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def safe_mkfilepath(filename: str):
    dir_name = os.path.abspath(os.path.join(filename, os.pardir))
    safe_mkdirs(dir_name)


def read_file_to_str(filename: str) -> str:
    """Read file and store into a string.

    Args:
        filename: the input file name.

    Return:
        The file content of input file.
    """
    filename = os.path.join(constants.ROOT_DIR, filename)
    with open(filename, 'r') as fin:
        content = fin.read()
    return content


def write_str_to_file(filename: str, content: str):
    """Write the string to a file.

    If the parent directory of the file dose not exist, create it firstly.

    Args:
        filename: the output file path.
        content: the content prepared to write to the file.
    """

    filename = os.path.join(constants.ROOT_DIR, filename)
    safe_mkfilepath(filename)
    with open(filename, 'w') as fout:
        fout.write(content)
