import numpy as np

import constants
from utils import file_utils
from utils.shell_args import SHELL_ARGS


def build_data_set(raw_data_path, write_prefix, target_idx):
    source_lines = list()
    target_lines = list()
    target_vocab = dict()
    lines = file_utils.read_file_to_str(raw_data_path).split('\n')
    for line in lines:
        columns = line.split(',')

        source_lines.append(','.join([word for idx, word in enumerate(columns) if idx != target_idx]))

        target_data = columns[target_idx]
        target_lines.append(target_data)
        if target_data not in target_vocab:
            target_vocab[target_data] = len(target_vocab)

    source_data = np.array(source_lines)
    target_data = np.array(target_lines)

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
    build_data_set(SHELL_ARGS.raw_data_path, SHELL_ARGS.prefix, 4)
