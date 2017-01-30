#!/bin/bash
bazel build evaluation_tools

../../../bazel-bin/magenta/models/basic_autofill_cnn/evaluation_tools \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128_bs=12,dataset=Piano-midi.de,in=2,lr=0.0625,num_i=0,num_pitches=88,rescale_loss=True,separate_instruments=False,use_residual=1,soft=False,corrupt=0.9,maskout_method=bernoulli \
--kind chordwise \
--fold valid \
--chronological True \
--num_crops 1 \
--crop_piece_len 64 \
--evaluation_batch_size 600 \ 
