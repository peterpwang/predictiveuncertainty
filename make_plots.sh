#/bin/bash

# Make NLL plot of CIFAR10
awk -F ',' '{print $1}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB2CIFAR10Model_result.tsv > results/EfficientNetB2CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_NLL.csv
paste -d ',' results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB2CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10Model_NLL.csv > results/NLL.csv
python3 make_plot.py results/NLL.csv NLL_CIFAR10 "upper right"
rm results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB2CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10Model_NLL.csv results/NLL.csv

# Make ECE plot of CIFAR10
awk -F ',' '{print $13}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB2CIFAR10Model_result.tsv > results/EfficientNetB2CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_ECE.csv
paste -d ',' results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB2CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10Model_ECE.csv > results/ECE.csv
python3 make_plot.py results/ECE.csv ECE_CIFAR10 "upper right"
rm results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB2CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10Model_ECE.csv results/ECE.csv

# Make Test error plot of CIFAR10
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB2CIFAR10Model_result.tsv > results/EfficientNetB2CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_Error.csv
paste -d ',' results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB2CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10Model_Error.csv > results/Error.csv
python3 make_plot.py results/Error.csv "TestError_CIFAR10" "upper right"
rm results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB2CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10Model_Error.csv results/Error.csv

# Make NLL plot of CIFAR100
awk -F ',' '{print $1}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB2CIFAR100Model_result.tsv > results/EfficientNetB2CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_NLL.csv
paste -d ',' results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB2CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100Model_NLL.csv > results/NLL.csv
python3 make_plot.py results/NLL.csv NLL_CIFAR100 "upper right"
rm results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB2CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100Model_NLL.csv results/NLL.csv

# Make ECE plot of CIFAR100
awk -F ',' '{print $13}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB2CIFAR100Model_result.tsv > results/EfficientNetB2CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_ECE.csv
paste -d ',' results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB2CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100Model_ECE.csv > results/ECE.csv
python3 make_plot.py results/ECE.csv ECE_CIFAR100 "upper right"
rm results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB2CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100Model_ECE.csv results/ECE.csv

# Make Test error plot of CIFAR100
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB2CIFAR100Model_result.tsv > results/EfficientNetB2CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_Error.csv
paste -d ',' results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB2CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100Model_Error.csv > results/Error.csv
python3 make_plot.py results/Error.csv "TestError_CIFAR100" "upper right"
rm results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB2CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100Model_Error.csv results/Error.csv

# Make NLL plot of MiniImagenet
awk -F ',' '{print $1}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB2MiniImagenetModel_result.tsv > results/EfficientNetB2MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_NLL.csv
paste -d ',' results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB2MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetModel_NLL.csv > results/NLL.csv
python3 make_plot.py results/NLL.csv NLL_MiniImagenet "upper right"
rm results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB2MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetModel_NLL.csv results/NLL.csv

# Make ECE plot of MiniImagenet
awk -F ',' '{print $13}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB2MiniImagenetModel_result.tsv > results/EfficientNetB2MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_ECE.csv
paste -d ',' results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB2MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetModel_ECE.csv > results/ECE.csv
python3 make_plot.py results/ECE.csv ECE_MiniImagenet "upper right"
rm results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB2MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetModel_ECE.csv results/ECE.csv

# Make Test error plot of MiniImagenet
awk -F ',' '{print (1-$15)}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB2MiniImagenetModel_result.tsv > results/EfficientNetB2MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_Error.csv
paste -d ',' results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB2MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetModel_Error.csv > results/Error.csv
python3 make_plot.py results/Error.csv "TestError_MiniImagenet" "upper right"
rm results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB2MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetModel_Error.csv results/Error.csv
