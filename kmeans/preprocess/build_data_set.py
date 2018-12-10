import os

import constants
from utils import file_utils
from utils.shell_args import SHELL_ARGS


def build_data_set(prefix):
    filename = os.path.join(prefix, '{}'.format(prefix))
    raw_filename = os.path.join(constants.RAW_DIR, filename)
    input_source_filename = os.path.join(constants.INPUT_DIR, '{}.{}'.format(filename, constants.SUFFIX_SOURCE_DATA))
    input_target_filename = os.path.join(constants.INPUT_DIR, '{}.{}'.format(filename, constants.SUFFIX_TARGET_DATA))

    lines = file_utils.read_file_to_str(raw_filename).split('\n')

    source_data = list()
    target_data = list()
    target_name2id = dict()
    for line in lines:
        columns = line.split(',')

        target_name = columns[0]
        if target_name not in target_name2id:
            target_id = len(target_name2id)
            target_name2id[target_name] = target_id
        else:
            target_id = target_name2id[target_name]
        target_data.append(str(target_id))

        source = ','.join(columns[1:])
        source_data.append(source)

    file_utils.write_str_to_file(input_source_filename, '\n'.join(source_data))
    file_utils.write_str_to_file(input_target_filename, '\n'.join(target_data))


if __name__ == '__main__':
    build_data_set(SHELL_ARGS.prefix)
