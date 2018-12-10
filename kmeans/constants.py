"""The constant values."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

RAW_DIR = os.path.join(STATIC_DIR, 'raw')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')

SUFFIX_SOURCE_DATA = 'source'
SUFFIX_TARGET_DATA = 'target'
