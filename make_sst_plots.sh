#/bin/bash

# Make NLL plot of SST
awk -F ',' '{print $1}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_NLL.csv
awk -F ',' '{print $1}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_NLL.csv
paste -d ',' results/TreeLSTMNet_NLL.csv results/TreeLSTMFocalNet_NLL.csv > results/NLL.csv
python3 make_sst_plot.py results/NLL.csv NLL_SST "upper left" 1
rm results/TreeLSTMNet_NLL.csv results/TreeLSTMFocalNet_NLL.csv results/NLL.csv

# Make ECE plot of SST
awk -F ',' '{print $13}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_ECE.csv
awk -F ',' '{print $13}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_ECE.csv
paste -d ',' results/TreeLSTMNet_ECE.csv results/TreeLSTMFocalNet_ECE.csv > results/ECE.csv
python3 make_sst_plot.py results/ECE.csv ECE_SST "best" 1
rm results/TreeLSTMNet_ECE.csv results/TreeLSTMFocalNet_ECE.csv results/ECE.csv

# Make Test error plot of SST
awk -F ',' '{print (1-$15)}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_Error.csv
awk -F ',' '{print (1-$15)}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_Error.csv
paste -d ',' results/TreeLSTMNet_Error.csv results/TreeLSTMFocalNet_Error.csv > results/Error.csv
python3 make_sst_plot.py results/Error.csv "TestError_SST" "upper right" 1
rm results/TreeLSTMNet_Error.csv results/TreeLSTMFocalNet_Error.csv results/Error.csv

