#!/bin/bash

source activate new_env3
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"
scripts_dir="scripts-final"


expt_num=3
dataset="pancreas"


config_path="$expt_num-$dataset.config"

# train and evaluate
# python -u train.py -c $scripts_dir/$config_path

# # evaluate only
python -u train.py -c $scripts_dir/$config_path --eval_model
