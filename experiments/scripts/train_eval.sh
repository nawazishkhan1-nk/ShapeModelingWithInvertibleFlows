#!/bin/bash

source activate new_env2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"

config_name = supershapes_2
python -u train.py -c scripts/$config_name.config | tee $config_name.txt
python -u train.py -c scripts/$config_name.config --eval_model
