#!/bin/bash
# Deploy code to macmini server and run evaluation

SERVER="macmini"
REMOTE_DIR="~/mft-downstream-task"

echo "Deploying code to $SERVER..."
# Sync specific files, excluding large/unnecessary artifacts
rsync -avz --exclude '.git' --exclude '__pycache__' \
    --exclude 'results' --exclude 'artifacts' \
    ./ $SERVER:$REMOTE_DIR

echo "Running evaluation on server..."
# Run evaluation for MFT Random Init
python evaluate_turblimp.py --model_path alibayram/mft-random-init --data_dir TurBLiMP/data/base --is_mft > mft_random_base_output.txt 2>&1 &

# Run evaluation for Tabi Random Init
python evaluate_turblimp.py --model_path alibayram/tabi-random-init --data_dir TurBLiMP/data/base > tabi_random_base_output.txt 2>&1 &

echo "Evaluations for random initialized models started in background."
echo "Check mft_random_base_output.txt and tabi_random_base_output.txt on server."
