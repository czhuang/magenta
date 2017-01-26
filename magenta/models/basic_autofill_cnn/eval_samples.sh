#!/bin/bash
bazel build eval_samples 
../../../bazel-bin/magenta/models/basic_autofill_cnn/eval_samples \
--generation_output_dir /data/lisatmp4/huangche/sigmoids  \
--separate_instruments False \
--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128_input_depth=2,learning_rate=0.0625,num_instruments=4,num_pitches=53,rescale_loss=True,separate_instruments=False,use_residual=1,use_softmax_loss=False,corrupt_ratio=0.9,maskout_method=bernoulli  \
