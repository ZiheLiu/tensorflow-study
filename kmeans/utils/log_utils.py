import logging
import os

import constants
from utils import file_utils
from utils.shell_args import SHELL_ARGS

formatter = '%(asctime)s - %(levelname)s - %(message)s'


data_name = SHELL_ARGS.prefix
logging_filename = os.path.join(constants.OUTPUT_DIR, data_name, 'logs')
file_utils.safe_mkfilepath(logging_filename)

logging.basicConfig(filename=logging_filename,
                    level=logging.INFO,
                    format=formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(formatter))
logging.getLogger().addHandler(console_handler)

LOGGER = logging
