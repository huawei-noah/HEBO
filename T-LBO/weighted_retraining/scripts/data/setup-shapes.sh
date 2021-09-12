
shapes_dir="weighted_retraining/data/shapes"
mkdir -p "$shapes_dir"
python weighted_retraining/shapes/squares_dataset.py \
    --grid_size=64 \
    --min_size=1 \
    --max_size=20 \
    --shuffle_seed=0 \
    --min_col=32 \
    --max_col=33 \
    --save_dir="$shapes_dir" \
    --num_replicates=10