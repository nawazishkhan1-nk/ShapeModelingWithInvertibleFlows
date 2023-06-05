#!/bin/bash

# source activate nonlinear
export OMP_NUM_THREADS=1
cd "/home/sci/iyerkrithika/ShapeWorks_new/ShapeModelingWithInvertibleFlows/experiments/"
scripts_dir="v2-scripts-final"


expt_num=2
dataset="pancreas"


config_path="$expt_num-$dataset-new.config"

# train 
python -u train.py -c $scripts_dir/$config_path

# Serialize Model only
# python -u train.py -c $scripts_dir/$config_path --serialize_model

# # evaluate only
python -u train.py -c $scripts_dir/$config_path --eval_model
