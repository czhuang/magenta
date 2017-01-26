#!/bin/bash
bazel build basic_autofill_cnn_train

../../../bazel-bin/magenta/models/basic_autofill_cnn/basic_autofill_cnn_train \
--run_dir /u/huangche/tf_logs_sigmoids \
--log_process True \
--dataset Piano-midi.de \
--separate_instruments False \
--num_instruments 0 \
--crop_piece_len 64 \
--model_name DeepStraightConvSpecs \
--num_layers 64 \
--num_filters 128 \
--use_residual 1 \
--batch_size 20 \
--maskout_method bernoulli \
--corrupt_ratio 0.90 \
--mask_indicates_context True \
--optimize_mask_only False \
--rescale_loss True \
--patience 10 \
--augment_by_transposing 0 \
--augment_by_halfing_doubling_durations 0 \
--denoise_mode False \
--num_epochs 0 \
--save_model_secs 30 \
