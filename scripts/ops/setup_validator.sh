#!/usr/bin/env bash
set -euo pipefail

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURRENT_USER="$(id -un)"
LOG_ROOT="$PROJECT_ROOT/logs"
PM2_LOG_DIR="$LOG_ROOT/pm2"
SYSTEMD_UNIT_PATH="$PROJECT_ROOT/scripts/systemd/sparket-validator.service"
ENV_FILE="$PROJECT_ROOT/.env"
YAML_FILE="$PROJECT_ROOT/sparket/config/sparket.yaml"
DEFAULT_DB_PORT="5435"
SUDO_CMD=""
DOCKER_GROUP_MODIFIED=0

if [[ $EUID -ne 0 ]]; then
  if sudo -n true >/dev/null 2>&1; then
    SUDO_CMD="sudo"
  else
    bold "This setup needs administrator privileges for package installation."
    sudo -v
    SUDO_CMD="sudo"
  fi
fi

ensure_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
  fi
}

ensure_env_var() {
  local key="$1"
  local value="$2"
  if [[ ! -f "$ENV_FILE" ]]; then
    touch "$ENV_FILE"
  fi
  if grep -Eq "^${key}=" "$ENV_FILE"; then
    return
  fi
  printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
}

apt_install() {
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found; skipping system package installation. Install dependencies manually."
    return
  fi
  info "Installing system packages via apt-get ..."
  $SUDO_CMD apt-get update
  $SUDO_CMD apt-get install -y build-essential git curl docker.io docker-compose-plugin nodejs npm
}

install_pm2() {
  if command -v pm2 >/dev/null 2>&1; then
    info "PM2 already installed."
    return
  fi
  if ! command -v npm >/dev/null 2>&1; then
    warn "npm not found; skipping PM2 installation."
    return
  fi
  info "Installing PM2 globally ..."
  $SUDO_CMD npm install -g pm2
}

add_user_to_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "docker not found; skipping docker group modification."
    return
  fi
  if groups "$CURRENT_USER" | grep -qw docker; then
    return
  fi
  info "Adding $CURRENT_USER to docker group ..."
  $SUDO_CMD usermod -aG docker "$CURRENT_USER"
  DOCKER_GROUP_MODIFIED=1
}

install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    info "Installing uv (Python toolchain) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    if [[ -f "$HOME/.cargo/env" ]]; then
      . "$HOME/.cargo/env"
    fi
  fi
  info "Ensuring Python 3.10 is available via uv ..."
  uv python install 3.10 >/dev/null 2>&1 || true
}

sync_python_deps() {
  info "Synchronising Python dependencies with uv ..."
  cd "$PROJECT_ROOT"
  uv sync --dev
}

setup_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    info "Creating baseline .env file ..."
    cat <<EOF >"$ENV_FILE"
# Sparket validator configuration
SPARKET_ROLE=validator
SPARKET_DATABASE__USER=sparket
SPARKET_DATABASE__PASSWORD=sparket
SPARKET_DATABASE__HOST=127.0.0.1
SPARKET_DATABASE__PORT=${DEFAULT_DB_PORT}
SPARKET_DATABASE__NAME=sparket
# DATABASE_URL is optional; the validator composes one from the fields above when missing.

PROJECT_ROOT=${PROJECT_ROOT}
PM2_LOG_DIR=${PM2_LOG_DIR}
VENV_PYTHON=${PROJECT_ROOT}/.venv/bin/python
EOF
  else
    ensure_env_var "SPARKET_ROLE" "validator"
    ensure_env_var "SPARKET_DATABASE__USER" "sparket"
    ensure_env_var "SPARKET_DATABASE__PASSWORD" "sparket"
    ensure_env_var "SPARKET_DATABASE__HOST" "127.0.0.1"
    ensure_env_var "SPARKET_DATABASE__PORT" "${DEFAULT_DB_PORT}"
    ensure_env_var "SPARKET_DATABASE__NAME" "sparket"
    ensure_env_var "PROJECT_ROOT" "${PROJECT_ROOT}"
    ensure_env_var "PM2_LOG_DIR" "${PM2_LOG_DIR}"
    ensure_env_var "VENV_PYTHON" "${PROJECT_ROOT}/.venv/bin/python"
  fi
}

setup_yaml_config() {
  if [[ -f "$YAML_FILE" ]]; then
    info "Existing sparket/config/sparket.yaml detected; leaving it untouched."
    return
  fi
  if [[ ! -f "$PROJECT_ROOT/sparket/config/sparket.example.yaml" ]]; then
    warn "Example config not found; skipping YAML generation."
    return
  fi
  info "Creating sparket/config/sparket.yaml with local defaults ..."
  ensure_dir "$PROJECT_ROOT/sparket/config"
  cat <<EOF >"$YAML_FILE"
role: validator

wallet:
  name: local-validator
  hotkey: default

subtensor:
  network: local
  chain_endpoint: ws://127.0.0.1:9945

chain:
  netuid: 2

database:
  host: 127.0.0.1
  port: ${DEFAULT_DB_PORT}
  user: sparket
  name: sparket
  pool_size: 50
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 1800
  echo: false
  bootstrap:
    auto_create: true
    admin_url: null
  docker:
    enabled: true
    image: postgres:18
    port: ${DEFAULT_DB_PORT}
    container_name: pg-sparket
    volume: pg_sparket_data
EOF
}

create_systemd_template() {
  ensure_dir "$(dirname "$SYSTEMD_UNIT_PATH")"
  if [[ -f "$SYSTEMD_UNIT_PATH" ]]; then
    info "systemd template already exists at $SYSTEMD_UNIT_PATH"
    return
  fi
  info "Writing systemd service template to $SYSTEMD_UNIT_PATH"
  cat <<EOF >"$SYSTEMD_UNIT_PATH"
[Unit]
Description=Sparket Validator
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_USER}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${PROJECT_ROOT}/.env
ExecStart=${PROJECT_ROOT}/.venv/bin/python sparket/entrypoints/validator.py
Restart=on-failure
RestartSec=5
StandardOutput=append:${LOG_ROOT}/sparket-validator.out.log
StandardError=append:${LOG_ROOT}/sparket-validator.err.log

[Install]
WantedBy=multi-user.target
EOF
}

generate_summary() {
  bold ""
  bold "Setup complete!"
  echo "Next steps:"
  echo "  1. Activate the virtual environment: source ${PROJECT_ROOT}/.venv/bin/activate"
  echo "  2. Run the validator once to bootstrap the database:"
  echo "       python sparket/entrypoints/validator.py"
  echo "  3. From ${PROJECT_ROOT}, start the validator under PM2:"
  echo "       pm2 start ecosystem.config.js --only validator-local"
  echo "       pm2 save"
  if (( DOCKER_GROUP_MODIFIED == 1 )); then
    warn "You were added to the docker group. Log out and log back in for this to take effect."
  fi
  echo ""
  echo "Optional: install the systemd service with"
  echo "  sudo cp ${SYSTEMD_UNIT_PATH} /etc/systemd/system/"
  echo "  sudo systemctl daemon-reload"
  echo "  sudo systemctl enable --now sparket-validator"
}

bold "Sparket validator automated setup"
info "Project root: ${PROJECT_ROOT}"
info "Running as user: ${CURRENT_USER}"

ensure_dir "$LOG_ROOT"
ensure_dir "$PM2_LOG_DIR"

apt_install
install_pm2
add_user_to_docker
install_uv
sync_python_deps
setup_env_file
setup_yaml_config
create_systemd_template

generate_summary
