#!/bin/bash

source activate shapeworks2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"
scripts_dir="v3-scripts-new"


expt_num=1
dataset="pancreas"


config_path="$expt_num-$dataset.config"

# train and evaluate
# python -u train.py -c $scripts_dir/$config_path

# Serialize Model only
# python -u train.py -c $scripts_dir/$config_path --serialize_model

# # evaluate only
python -u train.py -c $scripts_dir/$config_path --eval_model
