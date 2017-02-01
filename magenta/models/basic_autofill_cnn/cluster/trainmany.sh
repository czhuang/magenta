dataset=JSB_Chorales
for num_layers in 48 32 64; do
for num_filters in 96 160 128; do
for corrupt_ratio in 0.90 0.95 0.99; do
jobdispatch --gpu --mem=8G --raw="#PBS -l feature=k80" bash train.sh --num_layers $num_layers --num_filters $num_filters --corrupt_ratio $corrupt_ratio --dataset $dataset --maskout_method bernoulli
jobdispatch --gpu --mem=8G --raw="#PBS -l feature=k80" bash train.sh --num_layers $num_layers --num_filters $num_filters --corrupt_ratio $corrupt_ratio --dataset $dataset --maskout_method bernoulli --optimize_mask_only
jobdispatch --gpu --mem=8G --raw="#PBS -l feature=k80" bash train.sh --num_layers $num_layers --num_filters $num_filters --corrupt_ratio $corrupt_ratio --dataset $dataset --maskout_method bernoulli --rescale_loss=false
done
jobdispatch --gpu --mem=8G --raw="#PBS -l feature=k80" bash train.sh --num_layers $num_layers --num_filters $num_filters --dataset $dataset --maskout_method balanced_by_scaling
done
done

