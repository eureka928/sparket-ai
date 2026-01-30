# Sparket Custom Miner - Detailed Setup Guide

This guide walks you through every step to set up and run the Sparket custom miner from scratch.

---

## Part 1: Get a Server

### Option A: Hetzner Cloud (Recommended - Cheapest)

1. Go to https://www.hetzner.com/cloud
2. Create account and verify email
3. Click "Add Server"
4. Select:
   - **Location**: Nuremberg or Falkenstein (EU) or Ashburn (US)
   - **Image**: Ubuntu 24.04
   - **Type**: CX22 (2 vCPU, 4GB RAM) - €4.35/month
   - **Networking**: Public IPv4 (checked)
   - **SSH Key**: Add your SSH public key (see below)
5. Click "Create & Buy Now"

### Option B: DigitalOcean

1. Go to https://www.digitalocean.com
2. Create Droplet
3. Select:
   - **Image**: Ubuntu 24.04
   - **Plan**: Basic, Regular, $24/month (2 vCPU, 4GB)
   - **Region**: New York or San Francisco
   - **Authentication**: SSH Key
4. Create Droplet

### Generate SSH Key (if you don't have one)

On your local machine:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# View public key (copy this to server provider)
cat ~/.ssh/id_ed25519.pub
```

---

## Part 2: Connect to Server

### 2.1 SSH into Server

```bash
# Replace with your server's IP address
ssh root@YOUR_SERVER_IP
```

### 2.2 Create Miner User (Security Best Practice)

```bash
# Create new user
adduser miner

# Add to sudo group
usermod -aG sudo miner

# Switch to miner user
su - miner
```

### 2.3 Update System

```bash
# Update package lists
sudo apt update

# Upgrade all packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3-pip \
    jq \
    htop \
    tmux
```

---

## Part 3: Install Dependencies

### 3.1 Install UV (Python Package Manager)

```bash
# Download and install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (reload shell)
source ~/.bashrc

# Verify installation
uv --version
# Expected output: uv 0.x.x
```

### 3.2 Install Node.js and PM2

```bash
# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify Node.js
node --version
# Expected: v20.x.x

# Install PM2 globally
sudo npm install -g pm2

# Verify PM2
pm2 --version
# Expected: 5.x.x
```

### 3.3 Install Bittensor

```bash
# Install Bittensor CLI
pip3 install bittensor

# Verify installation
btcli --version
# Expected: bittensor x.x.x
```

---

## Part 4: Clone Repository

### 4.1 Clone Sparket Repository

```bash
# Go to home directory
cd ~

# Clone the repository (replace with actual repo URL)
git clone https://github.com/your-org/sparket-ai.git

# Enter directory
cd sparket-ai

# Check files
ls -la
```

### 4.2 Install Python Dependencies

```bash
# Install Python 3.10 via UV
uv python install 3.10

# Install all dependencies
uv sync --dev

# Verify virtual environment created
ls -la .venv/

# Activate virtual environment
source .venv/bin/activate

# Verify Python
python --version
# Expected: Python 3.10.x
```

---

## Part 5: Create Bittensor Wallet

### 5.1 Create Coldkey (Main Wallet)

```bash
# Create coldkey - THIS IS YOUR MAIN WALLET
btcli wallet new_coldkey --wallet.name miner
```

You will see:
```
IMPORTANT: Store this mnemonic in a secure (preferable offline place),
as anyone who has possession of this mnemonic can use it to regenerate
the key and access your tokens.

The mnemonic to the new coldkey is:

word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12

Specify password for key encryption:
```

**CRITICAL**:
- Write down the 12 words on paper
- Store in a safe place (not on computer)
- Never share with anyone
- Set a strong password

### 5.2 Create Hotkey (Mining Key)

```bash
# Create hotkey for mining operations
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

You will see another mnemonic - save this too (less critical than coldkey).

### 5.3 View Wallet Info

```bash
# View wallet overview
btcli wallet overview --wallet.name miner
```

Output shows:
```
Wallet Name: miner
Coldkey SS58: 5xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  <-- Your coldkey address
Hotkey SS58:  5yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy  <-- Your hotkey address
```

**Copy the Coldkey SS58 address** - you'll need this to receive TAO.

---

## Part 6: Fund Your Wallet

### 6.1 Get TAO

You need approximately **0.1 TAO** to register on the subnet.

**Option A: Buy from Exchange**
1. Create account on MEXC, Gate.io, or KuCoin
2. Buy TAO
3. Withdraw to your Coldkey SS58 address

**Option B: Testnet Faucet (Testing Only)**
```bash
# For testnet only
btcli wallet faucet --wallet.name miner --subtensor.network test
```

### 6.2 Check Balance

```bash
# Check mainnet balance
btcli wallet balance --wallet.name miner --subtensor.network finney

# Check testnet balance
btcli wallet balance --wallet.name miner --subtensor.network test
```

Wait until balance shows your TAO before proceeding.

---

## Part 7: Register on Subnet

### 7.1 Find Sparket Subnet ID

```bash
# List all subnets on mainnet
btcli subnet list --subtensor.network finney

# Look for "Sparket" and note the NETUID (example: 42)
```

### 7.2 Register Your Miner

```bash
# Replace <NETUID> with actual subnet number (e.g., 42)

# For MAINNET (Finney)
btcli subnet register \
    --netuid <NETUID> \
    --wallet.name miner \
    --wallet.hotkey default \
    --subtensor.network finney

# For TESTNET
btcli subnet register \
    --netuid <NETUID> \
    --wallet.name miner \
    --wallet.hotkey default \
    --subtensor.network test
```

You'll see:
```
Your balance is: τ0.100000000
The cost to register is: τ0.000001000
Do you want to continue? [y/n]: y
```

Type `y` and press Enter.

### 7.3 Verify Registration

```bash
# Check if you appear in the metagraph
btcli subnet metagraph --netuid <NETUID> --subtensor.network finney

# Look for your hotkey in the list
```

---

## Part 8: Configure Miner

### 8.1 Create Configuration File

```bash
cd ~/sparket-ai

# Copy example config
cp sparket/config/sparket.example.yaml sparket/config/sparket.yaml

# Edit config
nano sparket/config/sparket.yaml
```

Replace contents with:

```yaml
role: miner

chain:
  netuid: 42  # <-- Replace with actual NETUID
  endpoint: null

wallet:
  name: miner
  hotkey: default

subtensor:
  network: finney  # Use "test" for testnet

axon:
  host: 0.0.0.0
  port: 8094
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

### 8.2 Create Environment File

```bash
# Create .env file
nano .env
```

Add these contents:

```bash
# Wallet Configuration
BT_WALLET_NAME=miner
BT_WALLET_HOTKEY=default
SPARKET_NETUID=42

# Custom Miner - Enable
SPARKET_CUSTOM_MINER__ENABLED=true

# The-Odds-API Key (get from https://the-odds-api.com)
SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_api_key_here

# Odds Settings
SPARKET_CUSTOM_MINER__VIG=0.045

# Timing Settings
SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS=21600
SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS=7

# Rate Limits
SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE=60
SPARKET_CUSTOM_MINER__PER_MARKET_LIMIT_PER_MINUTE=10
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

### 8.3 Get The-Odds-API Key

1. Go to https://the-odds-api.com
2. Click "Get API Key"
3. Create free account
4. Copy your API key from dashboard
5. Paste into `.env` file (replace `your_api_key_here`)

---

## Part 9: Test Before Running

### 9.1 Seed Elo Ratings

```bash
cd ~/sparket-ai
source .venv/bin/activate

# Load environment variables
set -a && source .env && set +a

# Seed team ratings
uv run python -m sparket.miner.custom.data.seed_elo
```

Expected output:
```
Seeding Elo ratings for NFL, NBA, MLB, NHL...
Exported 124 ratings to /home/miner/.sparket/custom_miner/elo_ratings.json
```

### 9.2 Run Demo (No Network)

```bash
# Test without connecting to Bittensor
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

3. Generating odds for sample matchups...
------------------------------------------------------------

BUF @ KC (NFL)
  Elo Model:    KC 62.4% | BUF 37.6%
  Elo Odds:     1.55 / 2.51
...
```

### 9.3 Test API Connection

```bash
uv run python -m sparket.miner.custom.test_odds_api --sport NBA
```

Should show real NBA games with odds from 30+ bookmakers.

---

## Part 10: Run Miner with PM2

### 10.1 Create PM2 Configuration

```bash
cd ~/sparket-ai

# Create PM2 ecosystem file
nano ecosystem.config.js
```

Add these contents:

```javascript
module.exports = {
  apps: [{
    name: 'sparket-miner',
    script: '.venv/bin/python',
    args: '-m sparket.miner.custom.runner',
    cwd: '/home/miner/sparket-ai',
    interpreter: 'none',
    env: {
      PATH: '/home/miner/sparket-ai/.venv/bin:/usr/bin:/bin',
      BT_WALLET_NAME: 'miner',
      BT_WALLET_HOTKEY: 'default',
      SPARKET_NETUID: '42',
      SPARKET_CUSTOM_MINER__ENABLED: 'true',
      SPARKET_CUSTOM_MINER__ODDS_API_KEY: 'your_api_key_here',
      SPARKET_CUSTOM_MINER__VIG: '0.045',
    },
    max_memory_restart: '2G',
    restart_delay: 10000,
    autorestart: true,
    watch: false,
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    error_file: '/home/miner/sparket-ai/logs/error.log',
    out_file: '/home/miner/sparket-ai/logs/output.log',
  }]
};
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

**Important**: Replace `your_api_key_here` with your actual Odds API key and `42` with actual NETUID.

### 10.2 Create Logs Directory

```bash
mkdir -p ~/sparket-ai/logs
```

### 10.3 Start Miner

```bash
# Start the miner
pm2 start ecosystem.config.js

# Check status
pm2 status
```

Expected output:
```
┌─────┬─────────────────┬─────────────┬─────────┬─────────┬──────────┐
│ id  │ name            │ namespace   │ version │ mode    │ pid      │
├─────┼─────────────────┼─────────────┼─────────┼─────────┼──────────┤
│ 0   │ sparket-miner   │ default     │ N/A     │ fork    │ 12345    │
└─────┴─────────────────┴─────────────┴─────────┴─────────┴──────────┘
```

### 10.4 View Logs

```bash
# View live logs
pm2 logs sparket-miner

# View last 100 lines
pm2 logs sparket-miner --lines 100
```

### 10.5 Save PM2 Configuration

```bash
# Save current process list
pm2 save

# Setup auto-start on server reboot
pm2 startup

# Copy and run the command it shows (example):
# sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u miner --hp /home/miner
```

---

## Part 11: Monitoring

### 11.1 Check Miner Status

```bash
# PM2 status
pm2 status

# PM2 detailed info
pm2 show sparket-miner

# PM2 monitoring dashboard
pm2 monit
```

### 11.2 Check Bittensor Status

```bash
# Check your miner in metagraph
btcli subnet metagraph --netuid <NETUID> --subtensor.network finney

# Check wallet balance and emissions
btcli wallet overview --wallet.name miner --subtensor.network finney
```

### 11.3 Check System Resources

```bash
# CPU and Memory usage
htop

# Disk usage
df -h

# Network connections
netstat -tlnp
```

---

## Part 12: Common Commands

### Miner Management

```bash
# Stop miner
pm2 stop sparket-miner

# Restart miner
pm2 restart sparket-miner

# Delete from PM2
pm2 delete sparket-miner

# View logs
pm2 logs sparket-miner --lines 200
```

### Update Code

```bash
cd ~/sparket-ai

# Stop miner
pm2 stop sparket-miner

# Pull latest code
git pull origin main

# Update dependencies
uv sync --dev

# Restart miner
pm2 restart sparket-miner
```

### Backup Wallet

```bash
# Create backup
cd ~
tar -czvf wallet-backup-$(date +%Y%m%d).tar.gz .bittensor/wallets/miner

# Download to local machine (from your local terminal)
scp miner@YOUR_SERVER_IP:~/wallet-backup-*.tar.gz ./
```

---

## Part 13: Troubleshooting

### Problem: "Wallet not found"

```bash
# Check wallet exists
ls ~/.bittensor/wallets/

# If empty, create wallet again
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### Problem: "Not registered on subnet"

```bash
# Check balance
btcli wallet balance --wallet.name miner

# If balance is 0, fund wallet first
# Then register
btcli subnet register --netuid <NETUID> --wallet.name miner
```

### Problem: Miner crashes immediately

```bash
# Check logs for errors
pm2 logs sparket-miner --lines 500

# Common issues:
# - Missing API key: Add SPARKET_CUSTOM_MINER__ODDS_API_KEY to ecosystem.config.js
# - Wrong NETUID: Verify NETUID is correct
# - Wallet password needed: Use btcli to unlock
```

### Problem: "Connection refused" or timeout

```bash
# Check if Bittensor network is reachable
curl -s https://entrypoint-finney.opentensor.ai:443 && echo "OK"

# Check firewall
sudo ufw status
sudo ufw allow 8094/tcp  # Allow axon port
```

### Problem: High memory usage

```bash
# Check memory
free -h

# Reduce refresh interval (less frequent updates)
# Edit ecosystem.config.js:
SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS: '43200'  # 12 hours
```

---

## Part 14: Security Checklist

- [ ] Coldkey mnemonic written on paper and stored safely
- [ ] Hotkey mnemonic backed up
- [ ] Strong passwords set for wallet encryption
- [ ] SSH key authentication enabled
- [ ] Password authentication disabled for SSH
- [ ] Firewall configured (`ufw allow 22,8094`)
- [ ] Regular system updates scheduled
- [ ] Wallet files have correct permissions

```bash
# Set correct permissions
chmod 700 ~/.bittensor
chmod 600 ~/.bittensor/wallets/miner/coldkey
chmod 600 ~/.bittensor/wallets/miner/hotkeys/default
```

---

## Summary Checklist

- [ ] Server provisioned (Hetzner/DO/etc.)
- [ ] SSH access configured
- [ ] System updated
- [ ] UV installed
- [ ] PM2 installed
- [ ] Bittensor installed
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Coldkey created and backed up
- [ ] Hotkey created
- [ ] Wallet funded with TAO
- [ ] Registered on subnet
- [ ] Configuration files created
- [ ] Odds API key obtained
- [ ] Demo tested successfully
- [ ] PM2 configured and started
- [ ] Auto-restart enabled

---

*Your miner is now running! Check emissions with:*
```bash
btcli wallet overview --wallet.name miner
```

---

*Last updated: January 2026*
