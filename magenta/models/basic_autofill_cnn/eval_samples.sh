#!/bin/bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

bazel build eval_samples 

../../../bazel-bin/magenta/models/basic_autofill_cnn/eval_samples \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128_len=64,dataset=4part_Bach_chorales,sil=False,lr=0.0625,pad=True,patience=5,quant=0.5,rescale_loss=True,separate_instruments=True,pop=False,residual=1,soft=True,corrupt=0.9,maskout_method=balanced_by_scaling/ \
--kind imagewise \
--fold valid \
--chronological False \
--chronological_margin 0 \
--pitch_chronological False \
--num_crops 1 \
--crop_piece_len 0 \
--eval_len 0 \
--evaluation_batch_size 1000 \
--pad_mode none \
--use_pop_stats False \
--eval_test_mode False
