"""项目中用到的常量."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')

INPUT_IRIS_FILENAME = os.path.join(INPUT_DIR, 'iris.data')
INPUT_WINE_FILENAME = os.path.join(INPUT_DIR, 'wine.data')

# LABEL_0 = 'Iris-setosa'
# LABEL_1 = 'Iris-virginica'
IRIS_FEATURES = ('sepal length', 'sepal width', 'petal length', 'petal width')
WINE_FEATURES = ('Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                 'Hue', 'OD280/OD315 of diluted wines', 'Proline')

# hyper params
STOP_VALUE = 1e-6
LEARNING_RATE = 0.01
