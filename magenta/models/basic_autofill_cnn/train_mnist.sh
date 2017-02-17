#!/bin/bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

bazel build basic_autofill_cnn_train

../../../bazel-bin/magenta/models/basic_autofill_cnn/basic_autofill_cnn_train \
--run_dir /u/huangche/tf_logs_sigmoids \
--log_process True \
--dataset BinaryMNIST \
--separate_instruments False \
--num_instruments 0 \
--crop_piece_len 28 \
--model_name DeepStraightConvSpecs \
--num_layers 64 \
--num_filters 128 \
--start_filter_size 3 \
--use_residual 1 \
--batch_size 100 \
--maskout_method balanced_by_scaling \
--corrupt_ratio 0.90 \
--mask_indicates_context True \
--optimize_mask_only False \
--rescale_loss True \
--patience 5 \
--augment_by_transposing 0 \
--augment_by_halfing_doubling_durations 0 \
--denoise_mode False \
--num_epochs 0 \
--eval_freq 5 
--save_model_secs 30 \
--use_pop_stats True \
--pad False \

bazel build evaluation_tools

../../../bazel-bin/magenta/models/basic_autofill_cnn/evaluation_tools \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128-start_fs=3_len=28,dataset=BinaryMNIST,sil=False,lr=0.0625,pad=False,patience=5,quant=0.125,rescale=True,run_id=,sep=False,pop=True,res=1,soft=False,corrupt=0.9,mm=balanced_by_scaling/ \
--kind imagewise \
--fold valid \
--chronological False \
--chronological_margin 0 \
--pitch_chronological False \
--crop_piece_len 0 \
--eval_len 0 \
--num_crops 1 \
--evaluation_batch_size 1000 \
--pad_mode none \
--use_pop_stats True \
--eval_test_mode False
