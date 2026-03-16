#!/bin/bash
# ============================================================
# Deploy code to remote server (NO secrets, NO data)
# Usage: bash deploy/deploy.sh user@SERVER_IP
# ============================================================
set -e

SERVER="${1:?Usage: bash deploy/deploy.sh user@SERVER_IP}"
REMOTE_DIR="~/trader"

echo "=== Deploying to $SERVER ==="

rsync -avz --progress \
    --exclude 'config/' \
    --exclude 'data/' \
    --exclude 'charts/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'private.pem' \
    --exclude 'venv/' \
    --exclude '.git/' \
    --exclude 'files.zip' \
    --exclude '*.log' \
    --exclude '.tmp.driveupload/' \
    --exclude '*.bak' \
    --exclude 'python/' \
    ./ "${SERVER}:${REMOTE_DIR}/"

# Also sync the models CSV (needed for signal generation)
rsync -avz --progress \
    models/crypto_hourly_best_models.csv \
    "${SERVER}:${REMOTE_DIR}/models/"

echo ""
echo "=== Deployed! ==="
echo "Restart trader: ssh $SERVER 'sudo systemctl restart crypto-trader'"
