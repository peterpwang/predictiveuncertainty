#/bin/bash

START_TIME=$SECONDS
python3 $1 --epochs 50 --lr=0.001 $2 $3 $4
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo $1 " took " $ELAPSED_TIME >> run_log.txt

