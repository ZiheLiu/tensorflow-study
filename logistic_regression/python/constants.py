"""项目中用到的常量."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')

INPUT_IRIS_FILENAME = os.path.join(INPUT_DIR, 'iris.data')

LABEL_0 = 'Iris-setosa'
LABEL_1 = 'Iris-virginica'
FEATURE_0 = 2
FEATURE_1 = 3

# hyper params
STOP_VALUE = 1e-6
