#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)
# Change this to where data is loaded from.
data_dir="$code_dir/data"
# Path to where samples are saved.
generation_output_dir="samples"

# Path to trained model.
checkpoint="logs/straight-64-128_bs=10,corrupt=0.5,len=64,lr=0.0625,mm=orderless,num_i=4,n_pch=46,mask_only=False,quant=0.125,rescale=True,sep=True,res=True,soft=True/best_model.ckpt"

# Evaluation settings.
fold=valid
fold_index=1  # Optionally can specify index of specific piece to be evaluated.
unit=frame
chronological=false
ensemble_size=5  # Number of different orderings to average.

ipython --pdb -- "$code_dir"/evaluate.py \
--data_dir $data_dir \
--checkpoint $checkpoint \
--fold $fold \
--unit $unit \
--chronological $chronological \
--ensemble_size 5 \
#--fold_index $fold_index
