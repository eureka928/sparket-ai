#!/usr/bin/env bash
# Setup local validator + custom miner for testing
#
# Usage:
#   ./scripts/setup_local.sh
#
# This script:
#   1. Creates .env.local from template
#   2. Creates test wallets (local-validator, local-miner)
#   3. Shows next steps

set -euo pipefail

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*" >&2; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bold "=== Sparket Local Testing Setup ==="
echo ""

# Check prerequisites
if ! command -v docker &>/dev/null; then
  warn "Docker not found. Please install Docker first."
  warn "The validator needs Docker for PostgreSQL."
fi

if ! command -v pm2 &>/dev/null; then
  warn "PM2 not found. Install with: sudo npm install -g pm2"
fi

# Setup .env.local
if [[ ! -f .env.local ]]; then
  info "Creating .env.local from template..."
  cp .env.local.example .env.local
  echo "  Created .env.local"
  echo ""
  warn "Please edit .env.local and add your API keys:"
  echo "  - SDIO_API_KEY (required for validator market data)"
  echo "  - SPARKET_CUSTOM_MINER__ODDS_API_KEY (recommended for miner)"
else
  info ".env.local already exists, skipping."
fi

echo ""

# Create wallets
info "Checking for test wallets..."

VALIDATOR_WALLET="${SPARKET_VALIDATOR_WALLET:-local-validator}"
MINER_WALLET="${SPARKET_MINER_WALLET:-local-miner}"

create_wallet() {
  local name="$1"
  local wallet_path="$HOME/.bittensor/wallets/$name"

  if [[ -d "$wallet_path" ]]; then
    info "Wallet '$name' already exists."
    return 0
  fi

  info "Creating wallet '$name'..."
  if command -v btcli &>/dev/null; then
    btcli wallet create --wallet.name "$name" --wallet.hotkey default --no_password || {
      warn "Failed to create wallet. Create manually with:"
      echo "  btcli wallet create --wallet.name $name"
    }
  else
    warn "btcli not found. Create wallets manually:"
    echo "  btcli wallet create --wallet.name $name"
  fi
}

create_wallet "$VALIDATOR_WALLET"
create_wallet "$MINER_WALLET"

echo ""
bold "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env.local with your API keys:"
echo "   nano .env.local"
echo ""
echo "2. Start local subtensor (if using network=local):"
echo "   # Or change SPARKET_SUBTENSOR__NETWORK=test in .env.local"
echo ""
echo "3. Start the validator:"
echo "   pm2 start ecosystem.local.config.js --only local-validator"
echo "   pm2 logs local-validator"
echo ""
echo "4. Once validator is running, start the custom miner:"
echo "   pm2 start ecosystem.local.config.js --only local-custom-miner"
echo "   pm2 logs local-custom-miner"
echo ""
echo "5. Monitor both:"
echo "   pm2 monit"
echo ""
echo "6. Stop all:"
echo "   pm2 stop ecosystem.local.config.js"
echo ""
bold "Files created:"
echo "  - .env.local (configure this)"
echo "  - ecosystem.local.config.js (PM2 config)"
echo "  - sparket/config/sparket.local.yaml (validator YAML)"
