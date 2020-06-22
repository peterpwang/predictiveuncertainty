#/bin/bash

# Make NLL plot of 20 Newsgroups
awk -F ',' '{print $1}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_NLL.csv
awk -F ',' '{print $1}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_NLL.csv results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv > results/NLL.csv
python3 make_text_plot.py results/NLL.csv NLL_20Newsgroups "upper right" 1
rm results/PoolCNNTwentyNewsgroupsModel_NLL.csv results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv results/NLL.csv

# Make ECE plot of 20 Newsgroups
awk -F ',' '{print $13}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_ECE.csv
awk -F ',' '{print $13}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_ECE.csv results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv > results/ECE.csv
python3 make_text_plot.py results/ECE.csv ECE_20Newsgroups "best" 1
rm results/PoolCNNTwentyNewsgroupsModel_ECE.csv results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv results/ECE.csv

# Make Test error plot of 20 Newsgroups
awk -F ',' '{print (1-$15)}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_Error.csv
awk -F ',' '{print (1-$15)}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_Error.csv results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv > results/Error.csv
python3 make_text_plot.py results/Error.csv "TestError_20Newsgroups" "upper right" 1
rm results/PoolCNNTwentyNewsgroupsModel_Error.csv results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv results/Error.csv

