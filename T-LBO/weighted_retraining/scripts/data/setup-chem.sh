
# Copy stored data to correct folder
chem_dir="weighted_retraining/data/chem/zinc/orig_model"
mkdir -p "$chem_dir"
cp -r weighted_retraining/assets/data/chem_orig_model/*.txt "$chem_dir"

# Store directories to the correct scripts
preprocess_script="weighted_retraining/weighted_retraining/chem/preprocess_data.py"
logP_script="weighted_retraining/weighted_retraining/chem/calc_penalized_logP.py"
QED_script="weighted_retraining/weighted_retraining/chem/calc_qed.py"

# Normally you might make the vocab, but the vocab is already made
# (it was copied from the original repo,
# so may not be exactly reproducible with the code in this repo)
# To make vocab for another model/dataset, run a command like the following:
# python scripts/data/chem/create_vocab.py \
#   --input_file=data/chem/zinc/orig_model/train.txt \
#   --output_file=data/chem/zinc/orig_model/vocab-CHECK.txt

# Calculate penalized logP for all files
all_smiles_file="$chem_dir/all.txt"
cat "$chem_dir/train.txt" "$chem_dir/val.txt" > "$all_smiles_file"
python "$logP_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/pen_logP_all.pkl"

python "$QED_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/qed_all.pkl"

# Next, we preprocess all the data (takes a VERY long time sadly...)

# Training set
out_dir="$chem_dir"/tensors_train
mkdir "$out_dir"
python "$preprocess_script" -t "$chem_dir"/train.txt  -d "$out_dir" 

# Validation set
out_dir="$chem_dir"/tensors_val
mkdir "$out_dir"
python "$preprocess_script" -t "$chem_dir"/val.txt    -d "$out_dir" 

# Tiny training set (for testing)

out_dir="$chem_dir"/tensors_train_tiny
mkdir "$out_dir"
python "$preprocess_script"    -t "$chem_dir"/train.txt -d "$out_dir"     -e 1000

# Tiny validation set (also for testing)
out_dir="$chem_dir"/tensors_val_tiny
mkdir "$out_dir"
python "$preprocess_script" -t "$chem_dir"/val.txt -d "$out_dir"  -e 1000
