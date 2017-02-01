#for checkpoint_dir in logs_corrupt99_20170116/*; do
for checkpoint_dir in logs_balanced_20170117/*; do
jobdispatch --gpu --mem=8G bash eval.sh $checkpoint_dir
done

