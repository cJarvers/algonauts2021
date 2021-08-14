#!/bin/bash
# first step: feature extraction and PCA
python3 -m feature_extraction.generate_features_resnet -vdir /videos -sdir /mnt/features --ckpt /checkpoints/run_19689885/model_0.ckpt

# second step: train mapping to fmri data
for ROI in V1 V2 V3 V4 LOC FFA STS EBA PPA
do
    echo "--- predicting ROI $ROI ---"
    for s in sub01 sub02 sub03 sub04 sub05 sub06 sub07 sub08 sub09 sub10
    do
        echo "    predicting subject $s"
        for l in layer_1 layer_2 layer_3 layer_4 layer5
        do
            echo "        validating layer $l"
            python3 perform_encoding.py -rd /mnt/features/results -ad /mnt/features --model resnet3d50 -l $l --sub $s -r $ROI -m val -fd /fmri -v false -b 1000
        done
    done
done
