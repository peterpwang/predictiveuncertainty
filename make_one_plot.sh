#/bin/bash

# Make NLL plot of Resnet 50CIFAR 10
awk -F ',' '{print $1}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_NLL.csv
paste -d ',' results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Resnet50CIFAR10 "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/NLL.csv

# Make ECE plot of Resnet 50 CIFAR 10
awk -F ',' '{print $13}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_ECE.csv
paste -d ',' results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Resnet50CIFAR10 "best" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/ECE.csv

# Make Test error plot of Resnet 50 CIFAR 10
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_Error.csv
paste -d ',' results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Resnet50CIFAR10" "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Error.csv

# Make NLL plot of Resnet 50 CIFAR 100
awk -F ',' '{print $1}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_NLL.csv
paste -d ',' results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Resnet50CIFAR100 "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv results/NLL.csv

# Make ECE plot of Resnet 50 CIFAR 100
awk -F ',' '{print $13}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_ECE.csv
paste -d ',' results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Resnet50CIFAR100 "best" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv results/ECE.csv

# Make Test error plot of Resnet 50 CIFAR 100
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_Error.csv
paste -d ',' results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Resnet50CIFAR100" "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv results/Error.csv

# Make NLL plot of Resnet 50 Mini Imagenet
awk -F ',' '{print $1}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv
paste -d ',' results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Resnet50MiniImagenet "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/NLL.csv

# Make ECE plot of Resnet 50 Mini Imagenet
awk -F ',' '{print $13}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv
paste -d ',' results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Resnet50MiniImagenet "best" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/ECE.csv

# Make Test error plot of Resnet 50 Mini Imagenet
awk -F ',' '{print (1-$15)}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_Error.csv
paste -d ',' results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Resnet50MiniImagenet" "upper right" 1 "Resnet 50" "Resnet 50 (FL γ=1)"
rm results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Error.csv

# Make NLL plot of Densenet 121 CIFAR 10
awk -F ',' '{print $1}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_NLL.csv
paste -d ',' results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Densenet121CIFAR10 "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of Densenet 121 CIFAR 10
awk -F ',' '{print $13}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_ECE.csv
paste -d ',' results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Densenet121CIFAR10 "best" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of Densenet 121 CIFAR 10
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_Error.csv
paste -d ',' results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Densenet121CIFAR10" "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/Error.csv

# Make NLL plot of Densenet 121 CIFAR 100
awk -F ',' '{print $1}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_NLL.csv
paste -d ',' results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Densenet121CIFAR100 "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of Densenet 121 CIFAR 100
awk -F ',' '{print $13}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_ECE.csv
paste -d ',' results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Densenet121CIFAR100 "best" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of Densenet 121 CIFAR 100
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_Error.csv
paste -d ',' results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Densenet121CIFAR100" "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv results/Error.csv

# Make NLL plot of Densenet 121 Mini Imagenet
awk -F ',' '{print $1}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_NLL.csv
paste -d ',' results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_Densenet121MiniImagenet "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv results/NLL.csv

# Make ECE plot of Densenet 121 Mini Imagenet
awk -F ',' '{print $13}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_ECE.csv
paste -d ',' results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_Densenet121MiniImagenet "best" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv results/ECE.csv

# Make Test error plot of Densenet 121 Mini Imagenet
awk -F ',' '{print (1-$15)}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_Error.csv
paste -d ',' results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_Densenet121MiniImagenet" "upper right" 1 "Densenet 121" "Densenet 121 (FL γ=1)"
rm results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B0 CIFAR 10
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_NLL.csv
paste -d ',' results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB0CIFAR10FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB0CIFAR10 "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB0CIFAR10FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B0 CIFAR 10
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_ECE.csv
paste -d ',' results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB0CIFAR10FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB0CIFAR10 "best" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB0CIFAR10FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B0 CIFAR 10
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_Error.csv
paste -d ',' results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB0CIFAR10FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB0CIFAR10" "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB0CIFAR10FocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B0 CIFAR 100
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_NLL.csv
paste -d ',' results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB0CIFAR100FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB0CIFAR100 "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB0CIFAR100FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B0 CIFAR 100
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_ECE.csv
paste -d ',' results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB0CIFAR100FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB0CIFAR100 "best" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB0CIFAR100FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B0 CIFAR 100
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_Error.csv
paste -d ',' results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB0CIFAR100FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB0CIFAR100" "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB0CIFAR100FocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B0 Mini Imagenet
awk -F ',' '{print $1}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_NLL.csv
paste -d ',' results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB0MiniImagenetFocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB0MiniImagenet "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB0MiniImagenetFocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B0 Mini Imagenet
awk -F ',' '{print $13}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_ECE.csv
paste -d ',' results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB0MiniImagenetFocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB0MiniImagenet "best" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB0MiniImagenetFocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B0 Mini Imagenet
awk -F ',' '{print (1-$15)}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_Error.csv
paste -d ',' results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB0MiniImagenetFocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB0MiniImagenet" "upper right" 1 "EfficentNet B0" "EfficentNet B0 (FL γ=1)"
rm results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB0MiniImagenetFocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B7 CIFAR 10
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_NLL.csv
paste -d ',' results/EfficientNetB7CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB7CIFAR10 "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B7 CIFAR 10
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_ECE.csv
paste -d ',' results/EfficientNetB7CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB7CIFAR10 "best" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B7 CIFAR 10
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_Error.csv
paste -d ',' results/EfficientNetB7CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB7CIFAR10" "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10FocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B7 CIFAR 100
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_NLL.csv
paste -d ',' results/EfficientNetB7CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB7CIFAR100 "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100FocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B7 CIFAR 100
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_ECE.csv
paste -d ',' results/EfficientNetB7CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB7CIFAR100 "best" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100FocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B7 CIFAR 100
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_Error.csv
paste -d ',' results/EfficientNetB7CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB7CIFAR100" "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100FocalModel_Error.csv results/Error.csv

# Make NLL plot of EfficentNet B7 Mini Imagenet
awk -F ',' '{print $1}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_NLL.csv
awk -F ',' '{print $1}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_NLL.csv
paste -d ',' results/EfficientNetB7MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetFocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_EfficientNetB7MiniImagenet "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetFocalModel_NLL.csv results/NLL.csv

# Make ECE plot of EfficentNet B7 Mini Imagenet
awk -F ',' '{print $13}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_ECE.csv
awk -F ',' '{print $13}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_ECE.csv
paste -d ',' results/EfficientNetB7MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetFocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_EfficientNetB7MiniImagenet "best" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetFocalModel_ECE.csv results/ECE.csv

# Make Test error plot of EfficentNet B7 Mini Imagenet
awk -F ',' '{print (1-$15)}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_Error.csv
awk -F ',' '{print (1-$15)}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_Error.csv
paste -d ',' results/EfficientNetB7MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetFocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_EfficientNetB7MiniImagenet" "upper right" 1 "EfficentNet B7" "EfficentNet B7 (FL γ=1)"
rm results/EfficientNetB7MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetFocalModel_Error.csv results/Error.csv

# Make NLL plot of 20 Newsgroups
awk -F ',' '{print $1}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_NLL.csv
awk -F ',' '{print $1}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_NLL.csv results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_20Newsgroups "upper right" 1 "Pooling CNN" "Pooling CNN (FL γ=1)"
rm results/PoolCNNTwentyNewsgroupsModel_NLL.csv results/PoolCNNTwentyNewsgroupsFocalModel_NLL.csv results/NLL.csv

# Make ECE plot of 20 Newsgroups
awk -F ',' '{print $13}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_ECE.csv
awk -F ',' '{print $13}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_ECE.csv results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv > results/ECE.csv
python3 make_text_plot.py results/ECE.csv ECE_20Newsgroups "best" 1 "Pooling CNN" "Pooling CNN (FL γ=1)"
rm results/PoolCNNTwentyNewsgroupsModel_ECE.csv results/PoolCNNTwentyNewsgroupsFocalModel_ECE.csv results/ECE.csv

# Make Test error plot of 20 Newsgroups
awk -F ',' '{print (1-$15)}' results/PoolCNNTwentyNewsgroupsModel_result.tsv > results/PoolCNNTwentyNewsgroupsModel_Error.csv
awk -F ',' '{print (1-$15)}' results/PoolCNNTwentyNewsgroupsFocalModel_result.tsv > results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv
paste -d ',' results/PoolCNNTwentyNewsgroupsModel_Error.csv results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv > results/Error.csv
python3 make_text_plot.py results/Error.csv "TestError_20Newsgroups" "upper right" 1 "Pooling CNN" "Pooling CNN (FL γ=1)"
rm results/PoolCNNTwentyNewsgroupsModel_Error.csv results/PoolCNNTwentyNewsgroupsFocalModel_Error.csv results/Error.csv

# Make NLL plot of SST
awk -F ',' '{print $1}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_NLL.csv
awk -F ',' '{print $1}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_NLL.csv
paste -d ',' results/TreeLSTMNet_NLL.csv results/TreeLSTMFocalNet_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_SST "upper left" 1 "Tree LSTM" "Tree LSTM (FL γ=1)"
rm results/TreeLSTMNet_NLL.csv results/TreeLSTMFocalNet_NLL.csv results/NLL.csv

# Make ECE plot of SST
awk -F ',' '{print $13}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_ECE.csv
awk -F ',' '{print $13}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_ECE.csv
paste -d ',' results/TreeLSTMNet_ECE.csv results/TreeLSTMFocalNet_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_SST "best" 1 "Tree LSTM" "Tree LSTM (FL γ=1)"
rm results/TreeLSTMNet_ECE.csv results/TreeLSTMFocalNet_ECE.csv results/ECE.csv

# Make Test error plot of SST
awk -F ',' '{print (1-$15)}' results/TreeLSTMNet_result.tsv > results/TreeLSTMNet_Error.csv
awk -F ',' '{print (1-$15)}' results/TreeLSTMFocalNet_result.tsv > results/TreeLSTMFocalNet_Error.csv
paste -d ',' results/TreeLSTMNet_Error.csv results/TreeLSTMFocalNet_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_SST" "upper right" 1 "Tree LSTM" "Tree LSTM (FL γ=1)"
rm results/TreeLSTMNet_Error.csv results/TreeLSTMFocalNet_Error.csv results/Error.csv

