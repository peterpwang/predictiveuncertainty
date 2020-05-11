#/bin/bash

# Make NLL plot
awk -F ',' '{print $1}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_NLL.csv
paste -d ',' results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv > results/NLL.csv
python3 make_plot.py results/NLL.csv NLL "upper right"
rm results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv results/NLL.csv

# Make ECE plot
awk -F ',' '{print $13}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_ECE.csv
paste -d ',' results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv > results/ECE.csv
python3 make_plot.py results/ECE.csv ECE "upper right"
rm results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv results/ECE.csv

# Make Test error plot
awk -F ',' '{print $15}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_Error.csv
awk -F ',' '{print $15}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_Error.csv
awk -F ',' '{print $15}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_Error.csv
awk -F ',' '{print $15}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_Error.csv
awk -F ',' '{print $15}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_Error.csv
paste -d ',' results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv > results/Error.csv
python3 make_plot.py results/Error.csv "Test Error" "lower right"
rm results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv results/Error.csv
