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
ssh $SERVER "cd $REMOTE_DIR && \
    source ~/.zshrc && \
    python evaluate_turblimp.py --model_path alibayram/mft-downstream-task-embeddingmagibu --data_dir TurBLiMP/data/base --is_mft > mft_base_output.txt 2>&1 & \
    echo 'Evaluation started in background. Check mft_base_output.txt on server.'"

python evaluate_turblimp.py --model_path alibayram/mft-random-init --data_dir TurBLiMP/data/base --is_mft > ramdom_mft_base_output.txt && python evaluate_turblimp.py --model_path alibayram/tabi-random-init --data_dir TurBLiMP/data/base > random_tabi_base_output.txt