#!/bin/bash
source $HOME/tensorflow/bin/activate
export PYTHONPATH="$HOME/repos/magenta"

python $HOME/repos/magenta/magenta/models/basic_autofill_cnn/basic_autofill_cnn_train.py \
--run_dir /scratch/jvb-000-aa/cooijmat/run/coconet/tf_logs_sigmoids \
--input_dir /scratch/jvb-000-aa/cooijmat/datasets/huangche \
--log_process True \
--separate_instruments False \
--num_instruments 0 \
--crop_piece_len 64 \
--model_name DeepStraightConvSpecs \
--use_residual 1 \
--batch_size 20 \
--mask_indicates_context True \
--optimize_mask_only False \
--rescale_loss True \
--patience 10 \
--augment_by_transposing 0 \
--augment_by_halfing_doubling_durations 0 \
--denoise_mode False \
--num_epochs 0 \
--save_model_secs 30 \
"$@"

