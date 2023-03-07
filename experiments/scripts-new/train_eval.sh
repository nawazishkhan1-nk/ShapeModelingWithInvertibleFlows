#!/bin/bash

source activate new_env2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"

config_name="1-supershapes"
config_path="$config_name.config"
scripts_dir="scripts-new"

python -u train.py -c $scripts_dir/$config_path
python -u train.py -c $scripts_dir/$config_path --eval_model
