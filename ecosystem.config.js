/**
 * PM2 Ecosystem Configuration for Sparket Subnet
 * 
 * Usage:
 *   pm2 start ecosystem.config.js          # Start all processes
 *   pm2 start ecosystem.config.js --only validator-local  # Start specific process
 *   pm2 stop ecosystem.config.js           # Stop all processes
 *   pm2 restart ecosystem.config.js         # Restart all processes
 *   pm2 delete ecosystem.config.js          # Delete all processes
 *   pm2 logs validator-local               # View logs
 *   pm2 monit                              # Monitor processes
 * 
 * Note: This config automatically loads environment variables from .env file
 */

const fs = require('fs');
const path = require('path');

// Load .env file and parse environment variables
function loadEnvFile(envPath) {
  const env = {};
  try {
    if (fs.existsSync(envPath)) {
      const content = fs.readFileSync(envPath, 'utf8');
      const lines = content.split('\n');
      
      for (const line of lines) {
        // Skip comments and empty lines
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) {
          continue;
        }
        
        // Parse KEY=VALUE format
        const match = trimmed.match(/^([^=]+)=(.*)$/);
        if (match) {
          const key = match[1].trim();
          let value = match[2].trim();
          
          // Remove quotes if present
          if ((value.startsWith('"') && value.endsWith('"')) ||
              (value.startsWith("'") && value.endsWith("'"))) {
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

// Load .env file from project root
const defaultProjectRoot = process.env.PROJECT_ROOT || path.resolve(__dirname);
const envPath = path.join(defaultProjectRoot, '.env');
const envVars = loadEnvFile(envPath);

const projectRoot = envVars.PROJECT_ROOT || defaultProjectRoot;
const interpreter =
  envVars.VENV_PYTHON ||
  process.env.VENV_PYTHON ||
  path.join(projectRoot, '.venv', 'bin', 'python');
const scriptPath = path.join(projectRoot, 'sparket/entrypoints/validator.py');
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
  `PM2 will run from ${projectRoot}. Loaded ${Object.keys(envVars).length} environment variables from .env`
);

module.exports = {
  apps: [
    {
      name: 'validator-local',
      script: scriptPath,
      interpreter,
      cwd: projectRoot,
      instances: 1,
      exec_mode: 'fork',
      
      // Environment variables - merge .env file vars with defaults
      // .env file variables take precedence
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        SPARKET_ROLE: 'validator',
        PROJECT_ROOT: projectRoot,
        PM2_LOG_DIR: logDir,
        VENV_PYTHON: interpreter,
        SPARKET_AXON__HOST: envVars.SPARKET_AXON__HOST || '0.0.0.0',
        SPARKET_AXON__PORT: envVars.SPARKET_AXON__PORT || '8093',
        // Merge all .env variables
        ...envVars,
      },
      
      // Auto-restart configuration
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      
      // Restart behavior
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,
      
      // Logging
      error_file: path.join(logDir, 'validator-local-error.log'),
      out_file: path.join(logDir, 'validator-local-out.log'),
      log_file: path.join(logDir, 'validator-local-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      // Process management
      kill_timeout: 5000,
      wait_ready: false,
      listen_timeout: 10000,
      
      // Advanced options
      instance_var: 'NODE_APP_INSTANCE',
      pmx: true,
      automation: true,
      vizion: true,
    },
    
    // Auditor validator - disabled by default.
    // Enable by: pm2 start ecosystem.config.js --only auditor-local
    // Requires: SPARKET_AUDITOR__PRIMARY_HOTKEY and SPARKET_AUDITOR__PRIMARY_URL in .env
    {
      name: 'auditor-local',
      script: path.join(projectRoot, 'sparket/entrypoints/auditor.py'),
      interpreter,
      cwd: projectRoot,
      instances: 1,
      exec_mode: 'fork',
      
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        SPARKET_ROLE: 'auditor',
        PROJECT_ROOT: projectRoot,
        PM2_LOG_DIR: logDir,
        VENV_PYTHON: interpreter,
        ...envVars,
      },
      
      autorestart: true,
      watch: false,
      max_memory_restart: '512M',
      
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,
      
      error_file: path.join(logDir, 'auditor-local-error.log'),
      out_file: path.join(logDir, 'auditor-local-out.log'),
      log_file: path.join(logDir, 'auditor-local-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      kill_timeout: 5000,
      wait_ready: false,
      listen_timeout: 10000,
    },
  ],
  
  // Deployment configuration (optional)
  deploy: {
    // production: {
    //   user: 'deploy',
    //   host: 'your-server.com',
    //   ref: 'origin/main',
    //   repo: 'git@github.com:your-repo/sparket-subnet.git',
    //   path: '/var/www/sparket-subnet',
    //   'post-deploy': 'pip install -r requirements.txt && pm2 reload ecosystem.config.js',
    // },
  },
};

