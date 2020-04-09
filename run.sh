#/bin/bash

python3 $1 --epochs 150 --lr=0.1
python3 $1 --epochs 100 --lr=0.01 --resume
python3 $1 --epochs 100 --lr=0.001 --resume
