import numpy as np

import constants
from utils import file_utils
from utils.shell_args import SHELL_ARGS


def _get_target_data(target_data: str):
    target_data = float(target_data)
    if target_data <= 8:
        return '0'
    elif target_data <= 10:
        return '1'
    else:
        return '2'


def _get_source_data(columns: list, target_idx: int):
    item = list()
    for idx, word in enumerate(columns):
        if idx != target_idx:
            if idx == 0:
                item.append(1 if word == 'M' else 0)
            else:
                item.append(float(word))
    return item


def _normalization(source_data: np.ndarray):
    max_source_data = np.max(source_data, axis=0)
    min_source_data = np.min(source_data, axis=0)
    source_data = (source_data - min_source_data) / (max_source_data - min_source_data)
    return source_data


def build_data_set(raw_data_path, write_prefix, target_idx, delimiter=',', skip_head=False):
    source_lines = list()
    target_lines = list()
    target_vocab = dict()
    lines = file_utils.read_file_to_str(raw_data_path).split('\n')
    if skip_head:
        lines = lines[1:]
    for line in lines:
        columns = line.split(delimiter)
        target_data = _get_target_data(columns[target_idx])
        source_data = _get_source_data(columns, target_idx)
        source_lines.append(source_data)
        target_lines.append(target_data)

        if target_data not in target_vocab:
            target_vocab[target_data] = len(target_vocab)

    source_data = np.array(source_lines)
    target_data = np.array(target_lines)

    source_data = _normalization(source_data)

    source_data = np.array([','.join([str(word) for word in source_line]) for source_line in source_data])

    index_list = np.arange(len(source_data))
    np.random.shuffle(index_list)
    source_data = source_data[index_list]
    target_data = target_data[index_list]

    test_start_idx = int(len(source_data) * constants.TEST_RATE)
    eval_start_idx = int(len(source_data) * constants.EVAL_RATE)
    test_source_data = source_data[: test_start_idx]
    test_target_data = target_data[: test_start_idx]
    eval_source_data = source_data[test_start_idx: eval_start_idx]
    eval_target_data = target_data[test_start_idx: eval_start_idx]
    train_source_data = source_data[eval_start_idx:]
    train_target_data = target_data[eval_start_idx:]

    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_TRAIN_SOURCE_DATA),
                                 '\n'.join(train_source_data))
    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_TRAIN_TARGET_DATA),
                                 '\n'.join(train_target_data))

    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_EVAL_SOURCE_DATA),
                                 '\n'.join(eval_source_data))
    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_EVAL_TARGET_DATA),
                                 '\n'.join(eval_target_data))

    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_TEST_SOURCE_DATA),
                                 '\n'.join(test_source_data))
    file_utils.write_str_to_file("%s%s" % (write_prefix, constants.SUFFIX_TEST_TARGET_DATA),
                                 '\n'.join(test_target_data))

    file_utils.write_dict_to_json_file("%s%s" % (write_prefix, constants.SUFFIX_TARGET_VOCAB), target_vocab)


if __name__ == '__main__':
    build_data_set(SHELL_ARGS.raw_data_path, SHELL_ARGS.prefix, 8, delimiter=',', skip_head=False)
