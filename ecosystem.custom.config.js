/**
 * PM2 Configuration for Sparket Custom Miner (with Axon)
 *
 * Runs the FULL miner entrypoint with axon serving + custom ensemble model.
 * The base miner is disabled; the custom miner handles odds generation.
 *
 * Usage:
 *   1. Copy .env.custom.example to .env.custom and configure
 *   2. pm2 start ecosystem.custom.config.js
 *   3. pm2 logs custom-miner
 *   4. pm2 stop custom-miner
 *
 * Environment variables can be set in .env.custom or directly below.
 */

const fs = require('fs');
const path = require('path');

// Load environment from .env.custom file
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
          // Remove quotes
          if ((value.startsWith('"') && value.endsWith('"')) ||
              (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          env[key] = value;
        }
      }
      console.log(`Loaded ${Object.keys(env).length} vars from ${envPath}`);
    }
  } catch (error) {
    console.warn(`Could not load ${envPath}: ${error.message}`);
  }
  return env;
}

const projectRoot = path.resolve(__dirname);
const envPath = path.join(projectRoot, '.env.custom');
const envVars = loadEnvFile(envPath);

// Paths - uses the MAIN miner entrypoint (not custom runner)
const scriptPath = path.join(projectRoot, 'sparket', 'entrypoints', 'miner.py');
const interpreter = path.join(projectRoot, '.venv', 'bin', 'python');
const logDir = path.join(projectRoot, 'logs');

// Create log directory
try {
  fs.mkdirSync(logDir, { recursive: true });
} catch (error) {
  console.warn(`Could not create log dir: ${error.message}`);
}

module.exports = {
  apps: [
    {
      name: 'custom-miner',
      script: scriptPath,
      interpreter,
      cwd: projectRoot,

      // Environment variables
      env: {
        // Python settings
        PYTHONUNBUFFERED: '1',
        PYTHONDONTWRITEBYTECODE: '1',

        // ===== Miner Role =====
        SPARKET_ROLE: 'miner',

        // ===== REQUIRED: Wallet Configuration =====
        // These are used by the main miner entrypoint for axon + bittensor
        SPARKET_WALLET__NAME: envVars.SPARKET_WALLET__NAME || envVars.BT_WALLET_NAME || 'default',
        SPARKET_WALLET__HOTKEY: envVars.SPARKET_WALLET__HOTKEY || envVars.BT_WALLET_HOTKEY || 'default',

        // Also set BT_ variants for custom miner standalone fallback
        BT_WALLET_NAME: envVars.BT_WALLET_NAME || envVars.SPARKET_WALLET__NAME || 'default',
        BT_WALLET_HOTKEY: envVars.BT_WALLET_HOTKEY || envVars.SPARKET_WALLET__HOTKEY || 'default',

        // ===== REQUIRED: Network Configuration =====
        SPARKET_NETUID: envVars.SPARKET_NETUID || '57',

        // ===== REQUIRED: Axon Configuration =====
        // The axon is how validators communicate with this miner
        SPARKET_AXON__HOST: envVars.SPARKET_AXON__HOST || '0.0.0.0',
        SPARKET_AXON__PORT: envVars.SPARKET_AXON__PORT || '8094',

        // External IP/port (set if behind NAT or load balancer)
        // SPARKET_AXON__EXTERNAL_IP: envVars.SPARKET_AXON__EXTERNAL_IP || '',
        // SPARKET_AXON__EXTERNAL_PORT: envVars.SPARKET_AXON__EXTERNAL_PORT || '',

        // ===== Miner Selection: Custom ON, Base OFF =====
        SPARKET_CUSTOM_MINER__ENABLED: 'true',
        SPARKET_BASE_MINER__ENABLED: 'false',

        // ===== RECOMMENDED: The-Odds-API Key =====
        // Get free key at: https://the-odds-api.com/
        SPARKET_CUSTOM_MINER__ODDS_API_KEY: envVars.SPARKET_CUSTOM_MINER__ODDS_API_KEY || '',

        // ===== Model Configuration =====
        // Vig (margin) for odds calculation
        SPARKET_CUSTOM_MINER__VIG: envVars.SPARKET_CUSTOM_MINER__VIG || '0.045',

        // Engine weights for ensemble (should sum to ~1.0)
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__ELO: envVars.SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__ELO || '0.35',
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__MARKET: envVars.SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__MARKET || '0.55',
        SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__POISSON: envVars.SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__POISSON || '0.10',

        // ===== Timing Configuration =====
        SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS: envVars.SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS || '7',
        SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS: envVars.SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS || '21600',

        // ===== Calibration Configuration =====
        SPARKET_CUSTOM_MINER__CALIBRATION__ENABLED: envVars.SPARKET_CUSTOM_MINER__CALIBRATION__ENABLED || 'true',
        SPARKET_CUSTOM_MINER__CALIBRATION__MIN_SAMPLES: envVars.SPARKET_CUSTOM_MINER__CALIBRATION__MIN_SAMPLES || '30',

        // ===== Rate Limiting =====
        SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE: envVars.SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE || '10',
        SPARKET_CUSTOM_MINER__PER_MARKET_LIMIT_PER_MINUTE: envVars.SPARKET_CUSTOM_MINER__PER_MARKET_LIMIT_PER_MINUTE || '2',

        // ===== Outcome Detection =====
        SPARKET_CUSTOM_MINER__OUTCOME_CHECK_SECONDS: envVars.SPARKET_CUSTOM_MINER__OUTCOME_CHECK_SECONDS || '300',

        // ===== Control API =====
        SPARKET_MINER_API_ENABLED: envVars.SPARKET_MINER_API_ENABLED || 'true',
        SPARKET_MINER_API_PORT: envVars.SPARKET_MINER_API_PORT || '8198',

        // Pass through any additional env vars from .env.custom
        ...envVars,
      },

      // Logging
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: path.join(logDir, 'custom-miner-error.log'),
      out_file: path.join(logDir, 'custom-miner-out.log'),
      log_file: path.join(logDir, 'custom-miner-combined.log'),
      merge_logs: true,
      time: true,

      // Process management
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '30s',
      restart_delay: 5000,

      // Resource limits
      max_memory_restart: '2G',

      // Graceful shutdown
      kill_timeout: 10000,
      wait_ready: false,
      listen_timeout: 30000,
    },
  ],
};
