#!/bin/bash
bazel build gibbs 
../../../bazel-bin/magenta/models/basic_autofill_cnn/gibbs \
--generation_output_dir /data/lisatmp4/huangche/sigmoids  \
--separate_instruments False \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128_bs=12,dataset=Piano-midi.de,in=2,lr=0.0625,num_i=0,num_pitches=88,rescale_loss=True,separate_instruments=False,use_residual=1,soft=False,corrupt=0.9,maskout_method=bernoulli \
--num_steps 0 \
--temperature 1. \
--generation_type unconditioned \
--initialization sequential  \
--piece_length 32
