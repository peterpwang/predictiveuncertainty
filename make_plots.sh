#/bin/bash

# Make NLL plot of CIFAR10
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_NLL.csv
paste -d ',' results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB0CIFAR10FocalModel_NLL.csv results/EfficientNetB7CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_CIFAR10 "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)" 
rm results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Densenet121CIFAR10Model_NLL.csv results/Densenet121CIFAR10FocalModel_NLL.csv results/EfficientNetB0CIFAR10Model_NLL.csv results/EfficientNetB0CIFAR10FocalModel_NLL.csv results/EfficientNetB7CIFAR10Model_NLL.csv results/EfficientNetB7CIFAR10FocalModel_NLL.csv results/NLL.csv
echo "NLL_CIFAR10"

# Make ECE plot of CIFAR10
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_ECE.csv
paste -d ',' results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB0CIFAR10FocalModel_ECE.csv results/EfficientNetB7CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_CIFAR10 "best" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Densenet121CIFAR10Model_ECE.csv results/Densenet121CIFAR10FocalModel_ECE.csv results/EfficientNetB0CIFAR10Model_ECE.csv results/EfficientNetB0CIFAR10FocalModel_ECE.csv results/EfficientNetB7CIFAR10Model_ECE.csv results/EfficientNetB7CIFAR10FocalModel_ECE.csv results/ECE.csv
echo "ECE_CIFAR10"

# Make Test error plot of CIFAR10
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121CIFAR10Model_result.tsv > results/Densenet121CIFAR10Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121CIFAR10FocalModel_result.tsv > results/Densenet121CIFAR10FocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0CIFAR10Model_result.tsv > results/EfficientNetB0CIFAR10Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0CIFAR10FocalModel_result.tsv > results/EfficientNetB0CIFAR10FocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7CIFAR10Model_result.tsv > results/EfficientNetB7CIFAR10Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7CIFAR10FocalModel_result.tsv > results/EfficientNetB7CIFAR10FocalModel_Error.csv
paste -d ',' results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB0CIFAR10FocalModel_Error.csv results/EfficientNetB7CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_CIFAR10" "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Densenet121CIFAR10Model_Error.csv results/Densenet121CIFAR10FocalModel_Error.csv results/EfficientNetB0CIFAR10Model_Error.csv results/EfficientNetB0CIFAR10FocalModel_Error.csv results/EfficientNetB7CIFAR10Model_Error.csv results/EfficientNetB7CIFAR10FocalModel_Error.csv results/Error.csv
echo "Error_CIFAR10"

# Make NLL plot of CIFAR100
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_NLL.csv
paste -d ',' results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB0CIFAR100FocalModel_NLL.csv results/EfficientNetB7CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100FocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_CIFAR100 "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50CIFAR100Model_NLL.csv results/Resnet50CIFAR100FocalGamma1Model_NLL.csv results/Densenet121CIFAR100Model_NLL.csv results/Densenet121CIFAR100FocalModel_NLL.csv results/EfficientNetB0CIFAR100Model_NLL.csv results/EfficientNetB0CIFAR100FocalModel_NLL.csv results/EfficientNetB7CIFAR100Model_NLL.csv results/EfficientNetB7CIFAR100FocalModel_NLL.csv results/NLL.csv
echo "NLL_CIFAR100"

# Make ECE plot of CIFAR100
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_ECE.csv
paste -d ',' results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB0CIFAR100FocalModel_ECE.csv results/EfficientNetB7CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100FocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_CIFAR100 "best" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50CIFAR100Model_ECE.csv results/Resnet50CIFAR100FocalGamma1Model_ECE.csv results/Densenet121CIFAR100Model_ECE.csv results/Densenet121CIFAR100FocalModel_ECE.csv results/EfficientNetB0CIFAR100Model_ECE.csv results/EfficientNetB0CIFAR100FocalModel_ECE.csv results/EfficientNetB7CIFAR100Model_ECE.csv results/EfficientNetB7CIFAR100FocalModel_ECE.csv results/ECE.csv
echo "ECE_CIFAR100"

# Make Test error plot of CIFAR100
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR100Model_result.tsv > results/Resnet50CIFAR100Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR100FocalGamma1Model_result.tsv > results/Resnet50CIFAR100FocalGamma1Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121CIFAR100Model_result.tsv > results/Densenet121CIFAR100Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121CIFAR100FocalModel_result.tsv > results/Densenet121CIFAR100FocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0CIFAR100Model_result.tsv > results/EfficientNetB0CIFAR100Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0CIFAR100FocalModel_result.tsv > results/EfficientNetB0CIFAR100FocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7CIFAR100Model_result.tsv > results/EfficientNetB7CIFAR100Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7CIFAR100FocalModel_result.tsv > results/EfficientNetB7CIFAR100FocalModel_Error.csv
paste -d ',' results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB0CIFAR100FocalModel_Error.csv results/EfficientNetB7CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100FocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_CIFAR100" "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50CIFAR100Model_Error.csv results/Resnet50CIFAR100FocalGamma1Model_Error.csv results/Densenet121CIFAR100Model_Error.csv results/Densenet121CIFAR100FocalModel_Error.csv results/EfficientNetB0CIFAR100Model_Error.csv results/EfficientNetB0CIFAR100FocalModel_Error.csv results/EfficientNetB7CIFAR100Model_Error.csv results/EfficientNetB7CIFAR100FocalModel_Error.csv results/Error.csv
echo "Error_CIFAR100"

# Make NLL plot of MiniImagenet
awk -F ',' 'NR>1{print $6}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_NLL.csv
paste -d ',' results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB0MiniImagenetFocalModel_NLL.csv results/EfficientNetB7MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetFocalModel_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_MiniImagenet "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/Densenet121MiniImagenetModel_NLL.csv results/Densenet121MiniImagenetFocalModel_NLL.csv results/EfficientNetB0MiniImagenetModel_NLL.csv results/EfficientNetB0MiniImagenetFocalModel_NLL.csv results/EfficientNetB7MiniImagenetModel_NLL.csv results/EfficientNetB7MiniImagenetFocalModel_NLL.csv results/NLL.csv
echo "NLL_MiniImagenet"

# Make ECE plot of MiniImagenet
awk -F ',' 'NR>1{print $13}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_ECE.csv
paste -d ',' results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB0MiniImagenetFocalModel_ECE.csv results/EfficientNetB7MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetFocalModel_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_MiniImagenet "best" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/Densenet121MiniImagenetModel_ECE.csv results/Densenet121MiniImagenetFocalModel_ECE.csv results/EfficientNetB0MiniImagenetModel_ECE.csv results/EfficientNetB0MiniImagenetFocalModel_ECE.csv results/EfficientNetB7MiniImagenetModel_ECE.csv results/EfficientNetB7MiniImagenetFocalModel_ECE.csv results/ECE.csv
echo "ECE_MiniImagenet"

# Make Test error plot of MiniImagenet
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121MiniImagenetModel_result.tsv > results/Densenet121MiniImagenetModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Densenet121MiniImagenetFocalModel_result.tsv > results/Densenet121MiniImagenetFocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0MiniImagenetModel_result.tsv > results/EfficientNetB0MiniImagenetModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB0MiniImagenetFocalModel_result.tsv > results/EfficientNetB0MiniImagenetFocalModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7MiniImagenetModel_result.tsv > results/EfficientNetB7MiniImagenetModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/EfficientNetB7MiniImagenetFocalModel_result.tsv > results/EfficientNetB7MiniImagenetFocalModel_Error.csv
paste -d ',' results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB0MiniImagenetFocalModel_Error.csv results/EfficientNetB7MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetFocalModel_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_MiniImagenet" "upper right" 5 "Resnet 50" "Resnet 50(FL γ=1)" "Densenet 121" "Densenet 121(FL γ=1)" "EfficientNet B0" "EfficientNet B0(FL γ=1)" "EfficientNet B7" "EfficientNet B7(FL γ=1)"
rm results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Densenet121MiniImagenetModel_Error.csv results/Densenet121MiniImagenetFocalModel_Error.csv results/EfficientNetB0MiniImagenetModel_Error.csv results/EfficientNetB0MiniImagenetFocalModel_Error.csv results/EfficientNetB7MiniImagenetModel_Error.csv results/EfficientNetB7MiniImagenetFocalModel_Error.csv results/Error.csv
echo "Error_MiniImagenet"

# Make NLL plot of CIFAR10 and Mini Imagenet
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_NLL.csv
awk -F ',' 'NR>1{print $6}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv
paste -d ',' results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv > results/NLL.csv
python3 make_one_plot.py results/NLL.csv NLL_CIFAR10_MiniImagenet "upper right" 5 "Resnet 50 CIFAR 10" "Resnet 50(FL γ=1) CIFAR 10" "Resnet 50 Mini Imagenet" "Resnet 50(FL γ=1) Mini Imagenet" 
rm results/Resnet50CIFAR10Model_NLL.csv results/Resnet50CIFAR10FocalGamma1Model_NLL.csv results/Resnet50MiniImagenetModel_NLL.csv results/Resnet50MiniImagenetFocalGamma1Model_NLL.csv results/NLL.csv
echo "NLL_CIFAR10_MiniImagenet"

# Make ECE plot of CIFAR10 and Mini Imagenet
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_ECE.csv
awk -F ',' 'NR>1{print $13}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv
paste -d ',' results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv > results/ECE.csv
python3 make_one_plot.py results/ECE.csv ECE_CIFAR10_MiniImagenet "best" 5 "Resnet 50 CIFAR 10" "Resnet 50(FL γ=1) CIFAR 10" "Resnet 50 Mini Imagenet" "Resnet 50(FL γ=1) Mini Imagenet"
rm results/Resnet50CIFAR10Model_ECE.csv results/Resnet50CIFAR10FocalGamma1Model_ECE.csv results/Resnet50MiniImagenetModel_ECE.csv results/Resnet50MiniImagenetFocalGamma1Model_ECE.csv results/ECE.csv
echo "ECE_CIFAR10_MiniImagenet"

# Make Test error plot of CIFAR10 and Mini Imagenet
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR10Model_result.tsv > results/Resnet50CIFAR10Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50CIFAR10FocalGamma1Model_result.tsv > results/Resnet50CIFAR10FocalGamma1Model_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50MiniImagenetModel_result.tsv > results/Resnet50MiniImagenetModel_Error.csv
awk -F ',' 'NR>1{print (1-$15)}' results/Resnet50MiniImagenetFocalGamma1Model_result.tsv > results/Resnet50MiniImagenetFocalGamma1Model_Error.csv
paste -d ',' results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv > results/Error.csv
python3 make_one_plot.py results/Error.csv "TestError_CIFAR10_MiniImagenet" "upper right" 5 "Resnet 50 CIFAR 10" "Resnet 50(FL γ=1) CIFAR 10" "Resnet 50 Mini Imagenet" "Resnet 50(FL γ=1) Mini Imagenet"
rm results/Resnet50CIFAR10Model_Error.csv results/Resnet50CIFAR10FocalGamma1Model_Error.csv results/Resnet50MiniImagenetModel_Error.csv results/Resnet50MiniImagenetFocalGamma1Model_Error.csv results/Error.csv
echo "Error_CIFAR10_MiniImagenet"
