import json
import os

import constants


def safe_mkdirs(dir_name):
    """Create dir if the dir does not exist."""
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def check_and_create_dir(filename: str):
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
    check_and_create_dir(filename)
    with open(filename, 'w') as fout:
        fout.write(content)


def write_dict_to_json_file(filename: str, _dict: dict):
    """Write the dict to a JSON file.

    Args:
        filename: the JSON file path.
        _dict: the dict prepared to write to the JSON file.
    """
    filename = os.path.join(constants.ROOT_DIR, filename)
    check_and_create_dir(filename)
    with open(filename, 'w') as fout:
        json.dump(_dict, fout)


def read_json_file_to_dict(filename: str) -> dict:
    """Read JSON file and store the content to dict.

    Args:
        filename: the JSON file path.

    Return:
        The dict read from JSON file.
    """
    filename = os.path.join(constants.ROOT_DIR, filename)
    with open(filename, 'r') as fin:
        result = json.load(fin)
    return result
