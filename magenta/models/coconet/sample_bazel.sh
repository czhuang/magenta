#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)
# Change this to dir for saving experiment logs. 
log_dir="$code_dir/logs"

# Path to where samples are saved.
generation_output_dir="samples"

# Path to trained model.
#checkpoint="logs/straight-64-128_bs=2,corrupt=0.5,len=64,lr=0.0625,mm=orderless,num_i=4,n_pch=46,mask_only=False,quant=0.125,rescale=True,sep=True,res=True,soft=True/best_model.ckpt"
checkpoint=$1

# Generation parameters.
# Number of samples to generate in a batch.
gen_batch_size=2
piece_length=16
strategy=igibbs

# Run command.
bazel run -c opt --config=cuda --copt="-mavx" :sample \
-- \
--log_dir=$log_dir \
--checkpoint="$checkpoint" \
--gen_batch_size=$gen_batch_size \
--piece_length=$piece_length \
--temperature=0.99 \
--strategy=$strategy \
--generation_output_dir=$generation_output_dir \
--logtostderr


