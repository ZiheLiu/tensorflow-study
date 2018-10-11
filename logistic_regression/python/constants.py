"""项目中用到的常量."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')

INPUT_IRIS_FILENAME = os.path.join(INPUT_DIR, 'iris.data')

LABEL_0 = 'Iris-setosa'
LABEL_1 = 'Iris-virginica'
FEATURES = ('sepal length', 'sepal width', 'petal length', 'petal width')
FEATURE_0 = 1
FEATURE_1 = 2

# hyper params
STOP_VALUE = 1e-6
