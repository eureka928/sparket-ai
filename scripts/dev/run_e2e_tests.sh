#!/bin/bash
# Run E2E tests for Sparket subnet
#
# Usage:
#   ./scripts/dev/run_e2e_tests.sh                # Run all E2E tests
#   ./scripts/dev/run_e2e_tests.sh --infra        # Run only infrastructure tests
#   ./scripts/dev/run_e2e_tests.sh --scenarios    # Run scenario tests
#   ./scripts/dev/run_e2e_tests.sh --fast         # Skip slow tests (memory profiling)
#   ./scripts/dev/run_e2e_tests.sh --start        # Start nodes and run tests
#   ./scripts/dev/run_e2e_tests.sh --stop         # Stop E2E nodes
#   ./scripts/dev/run_e2e_tests.sh --full         # Full test suite with nodes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! docker ps &>/dev/null; then
        log_error "Docker not running"
        exit 1
    fi
    
    # Check postgres container
    if ! docker ps | grep -q pg-sparket; then
        log_error "pg-sparket container not running"
        exit 1
    fi
    
    # Check subtensor container
    if ! docker ps | grep -q local_chain; then
        log_error "local_chain (subtensor) container not running"
        exit 1
    fi
    
    # Check database exists
    if ! docker exec pg-sparket psql -U sparket -lqt | grep -q sparket_e2e; then
        log_warn "sparket_e2e database not found, creating..."
        docker exec pg-sparket psql -U sparket -c "CREATE DATABASE sparket_e2e;"
        
        # Run migrations
        log_info "Running migrations..."
        DATABASE_URL="postgresql+asyncpg://sparket:sparket@127.0.0.1:5435/sparket_e2e" \
            uv run sparket-alembic upgrade head
    fi
    
    log_info "Prerequisites OK"
}

# Start E2E nodes
start_nodes() {
    log_section "Starting E2E Nodes"
    
    # Check if already running
    if pm2 list 2>/dev/null | grep -q e2e-validator; then
        log_warn "E2E nodes already running, restarting..."
        pm2 restart ecosystem.e2e.config.js 2>/dev/null || pm2 start ecosystem.e2e.config.js
    else
        pm2 start ecosystem.e2e.config.js
    fi
    
    log_info "Waiting for nodes to initialize..."
    
    # Wait up to 60s for validator health
    for i in {1..60}; do
        if curl -s http://127.0.0.1:8199/health 2>/dev/null | grep -q '"status":"ok"'; then
            log_info "Validator healthy after ${i}s"
            break
        fi
        if [ $i -eq 60 ]; then
            log_warn "Validator may not be ready after 60s"
        fi
        sleep 1
    done
}

# Stop E2E nodes
stop_nodes() {
    log_section "Stopping E2E Nodes"
    pm2 stop ecosystem.e2e.config.js 2>/dev/null || true
    pm2 delete ecosystem.e2e.config.js 2>/dev/null || true
    log_info "Nodes stopped"
}

# Run tests
run_tests() {
    local test_path="${1:-tests/e2e/localnet/}"
    local extra_args="${2:-}"
    
    log_section "Running E2E Tests"
    log_info "Path: $test_path"
    
    source .venv/bin/activate
    python -m pytest "$test_path" -v --tb=short $extra_args
}

# Print summary
print_summary() {
    log_section "Test Summary"
    echo "
E2E Test Suite Components:
  - Infrastructure tests: Basic connectivity and config
  - Scenario tests:
    * OddsCompetitionScenario: CLV scoring
    * OutcomeVerificationScenario: PSS scoring
    * AdversarialScenario: Abuse resistance
    * EdgeCaseScenario: Boundary conditions
    * CrashRecoveryScenario: System resilience
    * MemoryProfilingScenario: Memory usage (slow)
  - Assertion tests: Scoring and weight invariants
"
}

# Parse arguments
case "${1:-}" in
    --infra)
        check_prerequisites
        run_tests "tests/e2e/localnet/test_infrastructure.py"
        ;;
    --scenarios)
        check_prerequisites
        run_tests "tests/e2e/localnet/test_scenarios.py"
        ;;
    --assertions)
        check_prerequisites
        run_tests "tests/e2e/localnet/test_assertions.py"
        ;;
    --fast)
        check_prerequisites
        run_tests "tests/e2e/localnet/" "-m 'not slow'"
        ;;
    --start)
        check_prerequisites
        start_nodes
        run_tests "tests/e2e/localnet/" "-m 'not slow'"
        ;;
    --full)
        check_prerequisites
        start_nodes
        run_tests "tests/e2e/localnet/"
        stop_nodes
        ;;
    --stop)
        stop_nodes
        ;;
    --summary)
        print_summary
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  (none)      Run all E2E tests (nodes must be running)"
        echo "  --infra     Run only infrastructure tests"
        echo "  --scenarios Run only scenario tests"
        echo "  --assertions Run only assertion tests"
        echo "  --fast      Run tests, skipping slow ones"
        echo "  --start     Start nodes and run fast tests"
        echo "  --full      Full suite: start nodes, run all tests, stop nodes"
        echo "  --stop      Stop E2E nodes"
        echo "  --summary   Print test suite summary"
        echo "  --help      Show this help"
        ;;
    *)
        check_prerequisites
        run_tests
        ;;
esac
