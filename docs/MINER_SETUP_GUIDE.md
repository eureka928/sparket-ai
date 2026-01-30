# Sparket Custom Miner - Server Setup Guide

## Server Requirements

### Minimum Specifications

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 2 cores | 4 cores |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 50 GB SSD | 100 GB SSD |
| **Network** | 100 Mbps | 1 Gbps |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |

### Cloud Provider Options

| Provider | Instance Type | Monthly Cost |
|----------|--------------|--------------|
| **AWS** | t3.medium | ~$30-40 |
| **DigitalOcean** | s-2vcpu-4gb | ~$24 |
| **Vultr** | vc2-2c-4gb | ~$24 |
| **Hetzner** | CX22 | ~$6 |
| **OVH** | VPS Starter | ~$7 |

> **Note**: Hetzner and OVH offer best value for Bittensor mining.

---

## Step 1: Server Setup

### 1.1 Initial Server Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget build-essential python3.10 python3.10-venv python3-pip

# Create miner user (optional but recommended)
sudo useradd -m -s /bin/bash miner
sudo usermod -aG sudo miner
sudo su - miner
```

### 1.2 Install UV (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
uv --version
```

### 1.3 Install PM2 (Process Manager)

```bash
# Install Node.js (required for PM2)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2
sudo npm install -g pm2
pm2 --version
```

---

## Step 2: Clone and Setup Repository

```bash
# Clone the repository
cd ~
git clone https://github.com/your-org/sparket-ai.git
cd sparket-ai

# Install Python dependencies
uv python install 3.10
uv sync --dev

# Activate virtual environment
source .venv/bin/activate
```

---

## Step 3: Bittensor Wallet Setup

### 3.1 Install Bittensor CLI

```bash
pip install bittensor
```

### 3.2 Create Wallet

```bash
# Create coldkey (main wallet - SECURE THIS!)
btcli wallet new_coldkey --wallet.name miner

# Create hotkey (for mining operations)
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# View wallet info
btcli wallet overview --wallet.name miner
```

> **IMPORTANT**:
> - Save your mnemonic phrases securely (offline, encrypted)
> - Never share your coldkey mnemonic
> - The hotkey is used for daily operations

### 3.3 Fund Your Wallet

You need TAO to register on the subnet:

1. **Get your coldkey address**:
   ```bash
   btcli wallet overview --wallet.name miner
   ```

2. **Transfer TAO** to your coldkey address from:
   - An exchange (MEXC, Gate.io, KuCoin)
   - Another wallet
   - Faucet (testnet only)

3. **Check balance**:
   ```bash
   btcli wallet balance --wallet.name miner
   ```

---

## Step 4: Register on Subnet

### 4.1 Find Sparket Subnet

```bash
# List all subnets
btcli subnet list

# Find Sparket subnet NETUID (example: 42)
```

### 4.2 Register Miner

```bash
# Register on mainnet (Finney)
btcli subnet register \
  --netuid <SPARKET_NETUID> \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney

# Or for testnet
btcli subnet register \
  --netuid <SPARKET_NETUID> \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network test
```

### 4.3 Verify Registration

```bash
btcli subnet metagraph --netuid <SPARKET_NETUID> --subtensor.network finney
```

---

## Step 5: Configuration

### 5.1 Create Configuration File

```bash
cp sparket/config/sparket.example.yaml sparket/config/sparket.yaml
```

Edit `sparket/config/sparket.yaml`:

```yaml
role: miner

chain:
  netuid: <SPARKET_NETUID>  # Replace with actual subnet ID
  endpoint: null  # Uses default based on network

wallet:
  name: miner
  hotkey: default

subtensor:
  network: finney  # or "test" for testnet

axon:
  host: 0.0.0.0  # Listen on all interfaces
  port: 8094     # Default miner port
```

### 5.2 Create Environment File

```bash
cp sparket/config/env.example .env
```

Edit `.env`:

```bash
# Bittensor wallet (can also be in YAML)
BT_WALLET_NAME=miner
BT_WALLET_HOTKEY=default
SPARKET_NETUID=<SPARKET_NETUID>

# Custom Miner Settings
SPARKET_CUSTOM_MINER__ENABLED=true
SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_odds_api_key_here
SPARKET_CUSTOM_MINER__VIG=0.045

# Timing (optional)
SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS=21600  # 6 hours
SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS=7

# Rate limits (optional)
SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE=60
```

### 5.3 Get The-Odds-API Key

1. Go to https://the-odds-api.com/
2. Sign up for free account (500 requests/month)
3. Copy your API key
4. Add to `.env` file

---

## Step 6: Test Before Production

### 6.1 Run Demo (No Network)

```bash
source .env
uv run python -m sparket.miner.custom.demo
```

Expected output:
```
============================================================
Custom Miner Demo (no network required)
============================================================
1. Initializing Elo engine with real ratings...
   Seeded 62 team ratings
2. Connecting to The-Odds-API...
   Connected!
...
```

### 6.2 Test API Connection

```bash
uv run python -m sparket.miner.custom.test_odds_api --sport NBA
```

---

## Step 7: Run Miner

### Option A: Direct Run (Testing)

```bash
source .env
uv run python -m sparket.miner.custom.runner \
  --wallet.name miner \
  --wallet.hotkey default
```

### Option B: PM2 (Production)

Create `ecosystem.custom.config.js`:

```javascript
module.exports = {
  apps: [{
    name: 'sparket-custom-miner',
    script: '.venv/bin/python',
    args: '-m sparket.miner.custom.runner',
    cwd: '/home/miner/sparket-ai',
    env: {
      BT_WALLET_NAME: 'miner',
      BT_WALLET_HOTKEY: 'default',
      SPARKET_NETUID: '<NETUID>',
      SPARKET_CUSTOM_MINER__ENABLED: 'true',
      SPARKET_CUSTOM_MINER__ODDS_API_KEY: 'your_key_here',
    },
    max_memory_restart: '2G',
    restart_delay: 10000,
    autorestart: true,
    watch: false,
  }]
};
```

Start with PM2:

```bash
pm2 start ecosystem.custom.config.js
pm2 save
pm2 startup  # Auto-start on reboot
```

### Option C: Systemd Service

Create `/etc/systemd/system/sparket-miner.service`:

```ini
[Unit]
Description=Sparket Custom Miner
After=network.target

[Service]
Type=simple
User=miner
WorkingDirectory=/home/miner/sparket-ai
Environment="PATH=/home/miner/sparket-ai/.venv/bin:/usr/bin"
EnvironmentFile=/home/miner/sparket-ai/.env
ExecStart=/home/miner/sparket-ai/.venv/bin/python -m sparket.miner.custom.runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sparket-miner
sudo systemctl start sparket-miner
sudo systemctl status sparket-miner
```

---

## Step 8: Monitoring

### 8.1 View Logs

```bash
# PM2 logs
pm2 logs sparket-custom-miner

# Systemd logs
journalctl -u sparket-miner -f

# Direct logs
tail -f ~/.pm2/logs/sparket-custom-miner-out.log
```

### 8.2 Check Miner Status

```bash
# Check if registered
btcli subnet metagraph --netuid <NETUID> --subtensor.network finney | grep $(cat ~/.bittensor/wallets/miner/hotkeys/default | jq -r '.ss58Address')

# Check emissions
btcli wallet overview --wallet.name miner
```

### 8.3 Monitor Performance

```bash
# System resources
htop

# PM2 monitoring
pm2 monit
```

---

## Step 9: Maintenance

### 9.1 Update Code

```bash
cd ~/sparket-ai
git pull origin main
uv sync --dev
pm2 restart sparket-custom-miner
```

### 9.2 Backup Wallet

```bash
# Backup wallet directory
tar -czvf wallet-backup-$(date +%Y%m%d).tar.gz ~/.bittensor/wallets/miner

# Store securely (encrypted, offline)
```

### 9.3 Monitor API Usage

```bash
# Check remaining API requests
uv run python -c "
import asyncio
from sparket.miner.custom.data.fetchers.odds_api import OddsAPIFetcher
import os

async def check():
    f = OddsAPIFetcher(api_key=os.getenv('SPARKET_CUSTOM_MINER__ODDS_API_KEY'))
    await f.get_odds('NBA')
    print(f'Remaining requests: {f.remaining_requests}')
    await f.close()

asyncio.run(check())
"
```

---

## Troubleshooting

### "Wallet not found"

```bash
# Check wallet exists
ls ~/.bittensor/wallets/

# Recreate if needed
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### "Not registered on subnet"

```bash
# Check registration
btcli subnet metagraph --netuid <NETUID>

# Re-register
btcli subnet register --netuid <NETUID> --wallet.name miner
```

### "Connection refused"

```bash
# Check if subtensor is reachable
curl -s https://entrypoint-finney.opentensor.ai:443

# Try alternative endpoint
export SUBTENSOR_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
```

### "Rate limited"

```bash
# Reduce submission frequency
export SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS=43200  # 12 hours
```

---

## Security Checklist

- [ ] Coldkey mnemonic backed up securely (offline)
- [ ] Server firewall configured (allow only 8094 for axon)
- [ ] SSH key authentication (disable password auth)
- [ ] Regular system updates
- [ ] Wallet files have restricted permissions (`chmod 600`)
- [ ] API keys in `.env` (not in code)
- [ ] Regular backups of `~/.sparket/custom_miner/` (ratings, calibration)

---

## Cost Estimation

| Item | Monthly Cost |
|------|-------------|
| Server (Hetzner CX22) | $6 |
| The-Odds-API (free tier) | $0 |
| TAO for registration | ~0.1 TAO (one-time) |
| **Total** | **~$6/month** |

---

## Support

- GitHub Issues: https://github.com/your-org/sparket-ai/issues
- Discord: [Sparket Discord Server]
- Bittensor Docs: https://docs.bittensor.com/

---

*Last updated: January 2026*
