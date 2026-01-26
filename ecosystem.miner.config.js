const fs = require('fs');
const path = require('path');

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
          if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          env[key] = value;
        }
      }
    }
  } catch (error) {
    console.error(`Error loading .env file: ${error.message}`);
  }
  return env;
}

const defaultProjectRoot = process.env.PROJECT_ROOT || path.resolve(__dirname);
const envPath = path.join(defaultProjectRoot, '.env');
const envVars = loadEnvFile(envPath);

const projectRoot = envVars.PROJECT_ROOT || defaultProjectRoot;
const interpreter =
  envVars.VENV_PYTHON ||
  process.env.VENV_PYTHON ||
  path.join(projectRoot, '.venv', 'bin', 'python');
const scriptPath = path.join(projectRoot, 'sparket/entrypoints/miner.py');
const logDir =
  envVars.PM2_LOG_DIR ||
  process.env.PM2_LOG_DIR ||
  path.join(projectRoot, 'sparket', 'logs', 'pm2');

try {
  fs.mkdirSync(logDir, { recursive: true });
} catch (error) {
  console.warn(`Unable to create PM2 log directory at ${logDir}: ${error.message}`);
}

console.log(
  `Miner PM2 will run from ${projectRoot}. Loaded ${Object.keys(envVars).length} environment variables from .env`
);

module.exports = {
  apps: [
    {
      name: 'miner-local',
      script: scriptPath,
      interpreter,
      cwd: projectRoot,
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        SPARKET_ROLE: 'miner',
        PROJECT_ROOT: projectRoot,
        PM2_LOG_DIR: logDir,
        VENV_PYTHON: interpreter,
        SPARKET_AXON__HOST: envVars.SPARKET_AXON__HOST || '0.0.0.0',
        SPARKET_AXON__PORT: envVars.SPARKET_AXON__PORT || '8094',
        SPARKET_MINER_CONFIG_FILE: envVars.SPARKET_MINER_CONFIG_FILE || path.join(projectRoot, 'sparket', 'config', 'miner.yaml'),
        SPARKET_CONFIG_FILE: envVars.SPARKET_CONFIG_FILE || path.join(projectRoot, 'sparket', 'config', 'miner.yaml'),
        SPARKET_ROLE: 'miner',
        ...envVars,
      },
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,
      error_file: path.join(logDir, 'miner-local-error.log'),
      out_file: path.join(logDir, 'miner-local-out.log'),
      log_file: path.join(logDir, 'miner-local-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      kill_timeout: 5000,
      wait_ready: false,
      listen_timeout: 10000,
      instance_var: 'NODE_APP_INSTANCE',
      pmx: true,
      automation: true,
      vizion: true,
    },
  ],
};

