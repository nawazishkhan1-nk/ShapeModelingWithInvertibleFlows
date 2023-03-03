#!/bin/bash

source activate new_env2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"

config_name="1-p-ldim-supershapes"
config_path="$config_name.config"
python -u train.py -c scripts/$config_path | tee $config_name.txt
python -u train.py -c scripts/$config_path --eval_model
