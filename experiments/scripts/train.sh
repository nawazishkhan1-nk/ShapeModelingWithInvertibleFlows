#!/bin/bash

source activate new_env2
export OMP_NUM_THREADS=1
cd "/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/experiments/"

python -u train.py -c scripts/supershapes_2.config | tee logNew2.txt
