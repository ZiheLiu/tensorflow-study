import json
import os

import constants


def check_and_create_dir(filename):
    dir_name = os.path.abspath(os.path.join(filename, os.pardir))
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def safe_mkdirs(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def read_file_to_str(filename):
    filename = os.path.join(constants.ROOT_DIR, filename)
    with open(filename, 'r') as fin:
        content = fin.read()
    return content


def write_str_to_file(filename, content):
    filename = os.path.join(constants.ROOT_DIR, filename)
    check_and_create_dir(filename)
    with open(filename, 'w') as fout:
        fout.write(content)


def write_dict_to_json_file(filename, _dict):
    filename = os.path.join(constants.ROOT_DIR, filename)
    check_and_create_dir(filename)
    with open(filename, 'w') as fout:
        json.dump(_dict, fout)


def read_json_file_to_dict(filename):
    filename = os.path.join(constants.ROOT_DIR, filename)
    with open(filename, 'r') as fin:
        result = json.load(fin)
    return result
