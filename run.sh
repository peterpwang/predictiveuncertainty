#/bin/bash

python3 $1 --epochs 15 --lr=0.1
python3 $1 --epochs 10 --lr=0.01 --resume
python3 $1 --epochs 10 --lr=0.01 --resume
