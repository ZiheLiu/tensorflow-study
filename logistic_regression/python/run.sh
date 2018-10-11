#!/usr/bin/env bash
python train.py --optimizer=newton --feature_0=0 --feature_1=1
python train.py --optimizer=newton --feature_0=0 --feature_1=2
python train.py --optimizer=newton --feature_0=0 --feature_1=3
python train.py --optimizer=newton --feature_0=1 --feature_1=2
python train.py --optimizer=newton --feature_0=1 --feature_1=3
python train.py --optimizer=newton --feature_0=2 --feature_1=3

python train.py --optimizer=gradient_descent --feature_0=0 --feature_1=1
python train.py --optimizer=gradient_descent --feature_0=0 --feature_1=2
python train.py --optimizer=gradient_descent --feature_0=0 --feature_1=3
python train.py --optimizer=gradient_descent --feature_0=1 --feature_1=2
python train.py --optimizer=gradient_descent --feature_0=1 --feature_1=3
python train.py --optimizer=gradient_descent --feature_0=2 --feature_1=3