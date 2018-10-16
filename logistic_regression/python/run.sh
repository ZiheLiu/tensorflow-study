#!/usr/bin/env bash
# for iris
#python train.py --optimizer=newton --data_type=iris --feature_0=0 --feature_1=1 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=newton --data_type=iris --feature_0=0 --feature_1=2 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=newton --data_type=iris --feature_0=0 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=newton --data_type=iris --feature_0=1 --feature_1=2 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=newton --data_type=iris --feature_0=1 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=newton --data_type=iris --feature_0=2 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=0 --feature_1=1 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=0 --feature_1=2 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=0 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=1 --feature_1=2 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=1 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#python train.py --optimizer=gradient_descent --data_type=iris --feature_0=2 --feature_1=3 --label_0=Iris-setosa --label_1=Iris-virginica
#
## for wine
#python train.py --optimizer=newton --data_type=wine --feature_0=1 --feature_1=2 --label_0=1 --label_1=2
#python train.py --optimizer=newton --data_type=wine --feature_0=5 --feature_1=10 --label_0=1 --label_1=2
#python train.py --optimizer=newton --data_type=wine --feature_0=4 --feature_1=9 --label_0=1 --label_1=2
#python train.py --optimizer=newton --data_type=wine --feature_0=6 --feature_1=11 --label_0=1 --label_1=2

python train.py --optimizer=gradient_descent --data_type=wine --feature_0=1 --feature_1=2 --label_0=1 --label_1=2
python train.py --optimizer=gradient_descent --data_type=wine --feature_0=5 --feature_1=10 --label_0=1 --label_1=2
python train.py --optimizer=gradient_descent --data_type=wine --feature_0=4 --feature_1=9 --label_0=1 --label_1=2
python train.py --optimizer=gradient_descent --data_type=wine --feature_0=6 --feature_1=11 --label_0=1 --label_1=2
