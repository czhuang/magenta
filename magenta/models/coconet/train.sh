#!/bin/bash

# Assumes that this script is run from within the Coconet root directory.  Change the following line if this is not the case.
code_dir=$(dirname $0)
# Change this to dir for saving experiment logs. 
log_dir="logs"
# Change this to where data is loaded from.
data_dir="$code_dir/testdata"
dataset=Jsb16thSeparated

# Data preprocessing.
crop_piece_len=64
separate_instruments=True
quantization_level=0.125  # 16th notes

# Hyperparameters.
maskout_method=orderless
num_layers=64
num_filters=128
batch_size=10

# Run command.
ipython --pdb -- "$code_dir"/train.py \
--log_dir $log_dir \
--log_process True \
--data_dir $data_dir \
--dataset $dataset \
--crop_piece_len $crop_piece_len \
--separate_instruments $separate_instruments \
--quantization_level $quantization_level \
--maskout_method $maskout_method \
--num_layers $num_layers \
--num_filters $num_filters \
--use_residual True \
--batch_size $batch_size \
