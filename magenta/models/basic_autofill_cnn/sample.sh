#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)
# Change this to dir for saving experiment logs. 
log_dir="$code_dir/logs"

# Path to where samples are saved.
generation_output_dir="$code_dir/samples"

# Path to trained model.
checkpoint_dir="trained_models/DeepStraightConvSpecs-64-128-start_fs=3_corrupt=0.5,len=128,dataset=bach-16th-priorwork-4_voices,sil=False,lr=0.0625,mm=balanced_by_scaling,mask_only=False,pad=True,patience=5,quant=0.125,rescale=True,run_id=,sep=True,pop=True,res=1,soft=True,"

# Generation parameters.
# Number of samples to generate in a batch.
gen_batch_size=32
piece_length=32
strategy=igibbs

# Run command.
python "$code_dir"/sample.py \
--log_dir $log_dir \
--checkpoint_dir "$checkpoint_dir" \
--gen_batch_size $gen_batch_size \
--piece_length $piece_length \
--temperature 0.99 \
--strategy $strategy \
--generation_output_dir $generation_output_dir 
