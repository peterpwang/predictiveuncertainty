#/bin/bash

START_TIME=$SECONDS
python3 $1 --epochs 150 --lr=0.1 $2 $3 $4
python3 $1 --epochs 100 --lr=0.01 --resume $2 $3 $4
python3 $1 --epochs 100 --lr=0.001 --resume $2 $3 $4
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo $1 " took " $ELAPSED_TIME >> run_log.txt

