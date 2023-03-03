#!/bin/bash

source activate shapeworks
export OMP_NUM_THREADS=1
cd ""

python -u train.py -c supershapes.config
