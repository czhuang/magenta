#!/bin/bash
source $HOME/tensorflow/bin/activate
export PYTHONPATH="$HOME/repos/magenta"

checkpoint_dir=$1
fold=valid

python $HOME/repos/magenta/magenta/models/basic_autofill_cnn/evaluation_tools.py \
--input_dir /scratch/jvb-000-aa/cooijmat/datasets/huangche \
--checkpoint_dir "$checkpoint_dir" --kind chordwise --fold $fold --crop_piece_len 64 \
--chronological True --chronological_margin 0 --num_crops 1 evaluation_batch_size 1000 \
2>&1 | tee "$checkpoint_dir"/evaluation_${fold}.txt

#--checkpoint_dir ~/tf_logs_sigmoids/DeepStraightConvSpecs-64-128_bs=12,dataset=Piano-midi.de,in=2,lr=0.0625,num_i=0,num_pitches=88,rescale_loss=True,separate_instruments=False,use_residual=1,soft=False,corrupt=0.9,maskout_method=bernoulli \

