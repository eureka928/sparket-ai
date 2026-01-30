#!/bin/bash
# Start the custom miner for production

set -e

echo "============================================"
echo "Starting Custom Miner"
echo "============================================"

# Check wallet exists
if ! btcli wallet list 2>/dev/null | grep -q "default"; then
    echo "ERROR: No wallet found. Create one with: btcli wallet new_coldkey"
    exit 1
fi

# Seed Elo ratings if not exists
ELO_PATH="$HOME/.sparket/custom_miner/elo_ratings.json"
if [ ! -f "$ELO_PATH" ]; then
    echo "Seeding Elo ratings..."
    uv run python -m sparket.miner.custom.data.seed_elo --output "$ELO_PATH"
fi

# Create logs directory
mkdir -p logs

# Start with PM2
echo "Starting miner with PM2..."
pm2 start ecosystem.custom.config.js

echo ""
echo "============================================"
echo "Custom miner started!"
echo "============================================"
echo ""
echo "Commands:"
echo "  pm2 logs custom-miner     # View logs"
echo "  pm2 status                # Check status"
echo "  pm2 stop custom-miner     # Stop miner"
echo "  pm2 restart custom-miner  # Restart miner"
echo ""
