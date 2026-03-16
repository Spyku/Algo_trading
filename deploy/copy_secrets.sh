#!/bin/bash
# ============================================================
# Copy secrets to remote server (one-time, or after key rotation)
# Usage: bash deploy/copy_secrets.sh user@SERVER_IP
#
# This copies your local config/ files to the server.
# These files are NEVER in git — they stay only on your machines.
# ============================================================
set -e

SERVER="${1:?Usage: bash deploy/copy_secrets.sh user@SERVER_IP}"
REMOTE_DIR="~/trader/config"

echo "=== Copying secrets to $SERVER ==="
echo "Files to copy:"
echo "  config/revolut_x_config.json  (API key)"
echo "  config/telegram_config.json   (bot token + chat ID)"
echo "  config/private.pem            (Ed25519 signing key)"
echo "  config/trading_config.json    (strategies + limits)"
echo ""

# Create remote config directory
ssh "$SERVER" "mkdir -p $REMOTE_DIR"

# Copy secret files
scp config/revolut_x_config.json "${SERVER}:${REMOTE_DIR}/"
scp config/telegram_config.json  "${SERVER}:${REMOTE_DIR}/"
scp config/private.pem           "${SERVER}:${REMOTE_DIR}/"
scp config/trading_config.json   "${SERVER}:${REMOTE_DIR}/"

# Lock down permissions on server
ssh "$SERVER" "chmod 600 ${REMOTE_DIR}/revolut_x_config.json ${REMOTE_DIR}/telegram_config.json ${REMOTE_DIR}/private.pem"

echo ""
echo "=== Secrets copied and locked down (chmod 600) ==="
echo ""
echo "Alternative: use .env file on server instead of JSON files."
echo "See .env.example for the format."
