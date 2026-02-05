/**
 * PM2 Ecosystem Configuration for Local Validator + Custom Miner Testing
 *
 * This config runs a validator locally for testing the custom miner.
 * Uses Docker PostgreSQL (auto-started) and local subtensor.
 *
 * Prerequisites:
 *   - Docker installed and running
 *   - Local subtensor running (or use test network)
 *   - Wallets created: local-validator, local-miner
 *
 * Setup:
 *   1. cp .env.local.example .env.local
 *   2. Edit .env.local with your settings (SDIO_API_KEY required for real data)
 *   3. Create wallets if needed:
 *      btcli wallet create --wallet.name local-validator
 *      btcli wallet create --wallet.name local-miner
 *
 * Usage:
 *   pm2 start ecosystem.local.config.js                    # Start all
 *   pm2 start ecosystem.local.config.js --only local-validator
 *   pm2 start ecosystem.local.config.js --only local-custom-miner
 *   pm2 logs local-validator                               # View logs
 *   pm2 logs local-custom-miner
 *   pm2 stop ecosystem.local.config.js                     # Stop all
 *   pm2 delete ecosystem.local.config.js                   # Delete all
 */

const fs = require('fs');
const path = require('path');

// Load .env.local file
function loadEnvFile(envPath) {
  const env = {};
  try {
    if (fs.existsSync(envPath)) {
      const content = fs.readFileSync(envPath, 'utf8');
      const lines = content.split('\n');

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;

        const match = trimmed.match(/^([^=]+)=(.*)$/);
        if (match) {
          const key = match[1].trim();
          let value = match[2].trim();

          if ((value.startsWith('"') && value.endsWith('"')) ||
              (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }

          env[key] = value;
        }
      }
    }
  } catch (error) {
    console.error(`Error loading env file: ${error.message}`);
  }
  return env;
}

const projectRoot = process.env.PROJECT_ROOT || path.resolve(__dirname);

// Try .env.local first, fall back to .env
let envPath = path.join(projectRoot, '.env.local');
if (!fs.existsSync(envPath)) {
  envPath = path.join(projectRoot, '.env');
}
const envVars = loadEnvFile(envPath);

const interpreter = envVars.VENV_PYTHON || path.join(projectRoot, '.venv', 'bin', 'python');
const logDir = envVars.PM2_LOG_DIR || path.join(projectRoot, 'sparket', 'logs', 'pm2-local');

try {
  fs.mkdirSync(logDir, { recursive: true });
} catch (error) {
  console.warn(`Unable to create log directory: ${error.message}`);
}

console.log(`PM2 LOCAL MODE - running from ${projectRoot}`);
console.log(`Env file: ${envPath}`);

// Base environment for local testing
const localEnvBase = {
  NODE_ENV: 'development',
  PYTHONUNBUFFERED: '1',
  PROJECT_ROOT: projectRoot,
  PM2_LOG_DIR: logDir,
  VENV_PYTHON: interpreter,

  // Network: use local subtensor by default, override via env
  SPARKET_SUBTENSOR__NETWORK: envVars.SPARKET_SUBTENSOR__NETWORK || 'local',
  SPARKET_CHAIN__ENDPOINT: envVars.SPARKET_CHAIN__ENDPOINT || 'ws://127.0.0.1:9945',

  // Database: Docker PostgreSQL on port 5433 (avoid conflicts)
  SPARKET_DATABASE__HOST: envVars.SPARKET_DATABASE__HOST || '127.0.0.1',
  SPARKET_DATABASE__PORT: envVars.SPARKET_DATABASE__PORT || '5433',
  SPARKET_DATABASE__USER: envVars.SPARKET_DATABASE__USER || 'sparket',
  SPARKET_DATABASE__PASSWORD: envVars.SPARKET_DATABASE__PASSWORD || 'sparket_local',
  SPARKET_DATABASE__NAME: envVars.SPARKET_DATABASE__NAME || 'sparket_local',
};

module.exports = {
  apps: [
    // =========================================================================
    // Local Validator
    // =========================================================================
    {
      name: 'local-validator',
      script: path.join(projectRoot, 'sparket/entrypoints/validator.py'),
      interpreter,
      cwd: projectRoot,
      instances: 1,
      exec_mode: 'fork',

      // Enable verbose logging
      args: '--logging.trace --logging.debug --logging.info',

      env: {
        ...localEnvBase,
        SPARKET_ROLE: 'validator',

        // Validator axon (receives miner submissions)
        SPARKET_AXON__HOST: '0.0.0.0',
        SPARKET_AXON__PORT: envVars.SPARKET_VALIDATOR_PORT || '8093',

        // Wallet
        SPARKET_WALLET__NAME: envVars.SPARKET_VALIDATOR_WALLET || 'local-validator',
        SPARKET_WALLET__HOTKEY: envVars.SPARKET_VALIDATOR_HOTKEY || 'default',

        // Netuid
        SPARKET_NETUID: envVars.SPARKET_NETUID || '1',

        // SportsDataIO API key (required for real market data)
        SDIO_API_KEY: envVars.SDIO_API_KEY || '',

        // Docker postgres config (auto-starts container)
        SPARKET_DATABASE__DOCKER__ENABLED: 'true',
        SPARKET_DATABASE__DOCKER__PORT: envVars.SPARKET_DATABASE__PORT || '5433',
        SPARKET_DATABASE__DOCKER__CONTAINER_NAME: 'pg-sparket-local',
        SPARKET_DATABASE__DOCKER__VOLUME: 'pg_sparket_local_data',

        // Bootstrap: auto-create database and run migrations
        SPARKET_DATABASE__BOOTSTRAP__AUTO_CREATE: 'true',

        // Merge any additional env vars
        ...envVars,
      },

      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,

      error_file: path.join(logDir, 'local-validator-error.log'),
      out_file: path.join(logDir, 'local-validator-out.log'),
      log_file: path.join(logDir, 'local-validator-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,

      kill_timeout: 5000,
    },

    // =========================================================================
    // Local Custom Miner
    // =========================================================================
    {
      name: 'local-custom-miner',
      script: path.join(projectRoot, 'sparket/entrypoints/miner.py'),
      interpreter,
      cwd: projectRoot,
      instances: 1,
      exec_mode: 'fork',

      // CLI args for bittensor
      args: '--logging.trace --logging.debug --logging.info',

      env: {
        ...localEnvBase,
        SPARKET_ROLE: 'miner',

        // Miner axon
        SPARKET_AXON__HOST: '0.0.0.0',
        SPARKET_AXON__PORT: envVars.SPARKET_MINER_PORT || '8094',

        // Wallet
        SPARKET_WALLET__NAME: envVars.SPARKET_MINER_WALLET || 'local-miner',
        SPARKET_WALLET__HOTKEY: envVars.SPARKET_MINER_HOTKEY || 'default',

        // Netuid
        SPARKET_NETUID: envVars.SPARKET_NETUID || '1',

        // Enable custom miner
        SPARKET_CUSTOM_MINER__ENABLED: 'true',
        SPARKET_BASE_MINER__ENABLED: 'false',

        // Custom miner settings
        SPARKET_CUSTOM_MINER__VIG: envVars.SPARKET_CUSTOM_MINER__VIG || '0.045',
        SPARKET_CUSTOM_MINER__ODDS_API_KEY: envVars.SPARKET_CUSTOM_MINER__ODDS_API_KEY || '',

        // Engine weights (Elo-heavy for originality)
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__ELO: '0.50',
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__MARKET: '0.35',
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__POISSON: '0.15',

        // Timing
        SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS: '21600',
        SPARKET_CUSTOM_MINER__TIMING__MIN_REFRESH_SECONDS: '300',

        // Merge additional env vars
        ...envVars,
      },

      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,

      error_file: path.join(logDir, 'local-custom-miner-error.log'),
      out_file: path.join(logDir, 'local-custom-miner-out.log'),
      log_file: path.join(logDir, 'local-custom-miner-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,

      kill_timeout: 5000,
    },
  ],
};
