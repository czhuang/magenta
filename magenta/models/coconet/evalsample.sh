#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)

# Path to where generated samples were stored.
sample_file="$code_dir/samples/sample_20170517233731_igibbs_DeepStraightConvSpecs_T0.99_l64_0.28min/final_pianorolls.npz"

# Path to trained model.
# FIXME checkpoint_dir flag is now called checkpoint and contains the path to the file
checkpoint_dir="trained_models/DeepStraightConvSpecs-64-128-start_fs=3_corrupt=0.5,len=128,dataset=bach-16th-priorwork-4_voices,sil=False,lr=0.0625,mm=balanced_by_scaling,mask_only=False,pad=True,patience=5,quant=0.125,rescale=True,run_id=,sep=True,pop=True,res=1,soft=True,"

# Evaluation settings.
#fold_index=  # Optionally can specify index of specific piece to be evaluated.
unit=frame
chronological=false
ensemble_size=5  # Number of different orderings to average.


python "$code_dir"/evaluate_npz.py $sample_file \
--checkpoint_dir $checkpoint_dir \
--unit $unit \
--chronological $chronological \
--ensemble_size 5 \
#--fold_index $fold_index
