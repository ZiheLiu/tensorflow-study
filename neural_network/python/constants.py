"""The constant values."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')

TEST_RATE = 0.10
EVAL_RATE = 0.10 + TEST_RATE

SUFFIX_TRAIN_SOURCE_DATA = '.train.source'
SUFFIX_TRAIN_TARGET_DATA = '.train.target'
SUFFIX_EVAL_SOURCE_DATA = '.eval.source'
SUFFIX_EVAL_TARGET_DATA = '.eval.target'
SUFFIX_TEST_SOURCE_DATA = '.test.source'
SUFFIX_TEST_TARGET_DATA = '.test.target'
SUFFIX_TARGET_VOCAB = '.target.vocab.json'

# hyper params
LEARNING_RATE = 0.01
HIDDEN_SIZE = 32
BATCH_SIZE = 8
EPOCHS = 300
