#!/bin/bash
# ============================================================
# ONE-TIME Oracle Cloud server setup for crypto auto-trader
# Run: bash setup_server.sh
# ============================================================
set -e

TRADER_DIR="$HOME/trader"

echo "=== Crypto Trader — Server Setup ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.13
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.13 python3.13-venv python3.13-dev

# Create project structure
mkdir -p "$TRADER_DIR"/{config,models,data,charts}

# Create venv
python3.13 -m venv "$TRADER_DIR/venv"
source "$TRADER_DIR/venv/bin/activate"
pip install --upgrade pip

# Install requirements (copy requirements.txt first via deploy.sh)
if [ -f "$TRADER_DIR/requirements.txt" ]; then
    pip install -r "$TRADER_DIR/requirements.txt"
else
    echo "WARNING: requirements.txt not found. Run deploy.sh first, then re-run this."
fi

# Create systemd service for auto-start + auto-restart
sudo tee /etc/systemd/system/crypto-trader.service > /dev/null << EOF
[Unit]
Description=Crypto Auto Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$TRADER_DIR
ExecStart=$TRADER_DIR/venv/bin/python crypto_revolut_trader.py --loop
Restart=on-failure
RestartSec=60
StandardOutput=append:$TRADER_DIR/trader.log
StandardError=append:$TRADER_DIR/trader.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crypto-trader

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. From your PC, run:  bash deploy/deploy.sh user@SERVER_IP"
echo "  2. From your PC, run:  bash deploy/copy_secrets.sh user@SERVER_IP"
echo "  3. Start trader:       sudo systemctl start crypto-trader"
echo "  4. View logs:          journalctl -u crypto-trader -f"
echo "  5. Or tail log file:   tail -f ~/trader/trader.log"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status crypto-trader    # check status"
echo "  sudo systemctl restart crypto-trader   # restart after deploy"
echo "  sudo systemctl stop crypto-trader      # stop"
