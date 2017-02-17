#!/bin/bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

bazel build basic_autofill_cnn_train

../../../bazel-bin/magenta/models/basic_autofill_cnn/basic_autofill_cnn_train \
--run_dir /u/huangche/tf_logs_sigmoids \
--log_process True \
--input_dir /data/lisatmp4/huangche/data/ \
--dataset bach-16th-priorwork-4_voices \
--separate_instruments True \
--num_instruments 4 \
--crop_piece_len 64 \
--model_name DeepStraightConvSpecs \
--num_layers 64 \
--num_filters 128 \
--use_residual 1 \
--batch_size 20 \
--maskout_method balanced_by_scaling \
--corrupt_ratio 0.90 \
--mask_indicates_context True \
--optimize_mask_only False \
--rescale_loss True \
--patience 10 \
--augment_by_transposing 0 \
--augment_by_halfing_doubling_durations 0 \
--denoise_mode False \
--num_epochs 0 \
--eval_freq 5 \
--save_model_secs 30 \
--use_pop_stats True \
--quantization_level 0.125 \
--pad True \

bazel build evaluation_tools

../../../bazel-bin/magenta/models/basic_autofill_cnn/evaluation_tools \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128-start_fs=3_len=64,dataset=bach-16th-priorwork-4_voices,sil=False,lr=0.0625,pad=True,patience=5,quant=0.125,rescale=True,run_id=,sep=True,pop=False,res=1,soft=True,corrupt=0.9,mm=balanced_by_scaling/ \
--kind chordwise \
--fold valid \
--chronological True \
--chronological_margin 0 \
--pitch_chronological False \
--num_crops 1 \
--crop_piece_len 0 \
--eval_len 0 \
--evaluation_batch_size 100 \
--pad_mode wrap \
--use_pop_stats True \
--eval_test_mode False
