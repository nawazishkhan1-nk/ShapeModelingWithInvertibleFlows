#!/bin/bash

source activate new_env2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"

config_name="1-ldim-multi-supershapes"
config_path="$config_name.config"
for i in {32..119..5}
do
    echo "Training $i latent dim model..."
    python -u train.py -c scripts/$config_path --modellatentdim  $i -i $i | tee log$i.txt
    echo "$i latent dim model Trained"
    python -u train.py -c scripts/$config_path --eval_model -i $i
done

