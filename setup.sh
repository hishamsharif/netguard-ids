#!/usr/bin/env bash
# =============================================================================
# NetGuard IDS — Setup Script
# Usage: bash setup.sh [--docker-only | --local-only | --skip-pg]
#
# What it does:
#   1. Verify / install Miniconda
#   2. Create/update the dl_env conda environment
#   3. Install Python dependencies
#   4. Optionally set up PostgreSQL (local dev)
#   5. Build Docker images (unless --local-only)
#   6. Create .env from template if missing
#   7. Create required runtime directories
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="dl_env"
DOCKER_ONLY=false
LOCAL_ONLY=false
WITH_PG=false   # PostgreSQL local setup is opt-in; default DB is SQLite

for arg in "$@"; do
  case "$arg" in
    --docker-only) DOCKER_ONLY=true ;;
    --local-only)  LOCAL_ONLY=true  ;;
    --with-pg)     WITH_PG=true     ;;
  esac
done

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  [OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}  [WARN]${NC} $*"; }
fail() { echo -e "${RED}  [FAIL]${NC} $*"; exit 1; }
step() { echo -e "\n${CYAN}▶ $*${NC}"; }

echo -e "${CYAN}"
echo "  ╔════════════════════════════════════════╗"
echo "  ║   NetGuard IDS — Environment Setup    ║"
echo "  ╚════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# 1. Conda / Miniconda
# =============================================================================
step "Checking conda"
if ! command -v conda &>/dev/null; then
  warn "conda not found — installing Miniconda"
  MINICONDA_SH="$HOME/miniconda_install.sh"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    -o "$MINICONDA_SH"
  bash "$MINICONDA_SH" -b -p "$HOME/miniconda"
  rm "$MINICONDA_SH"
  export PATH="$HOME/miniconda/bin:$PATH"
  conda init bash 2>/dev/null || true
  source "$HOME/.bashrc" 2>/dev/null || true
fi
ok "conda $(conda --version)"

# =============================================================================
# 2. Conda environment
# =============================================================================
step "Conda environment: $CONDA_ENV"
if conda env list | grep -q "^${CONDA_ENV} "; then
  ok "env '$CONDA_ENV' exists"
else
  warn "Creating env '$CONDA_ENV' (Python 3.12)"
  conda create -y -n "$CONDA_ENV" python=3.12
  ok "env created"
fi

# =============================================================================
# 3. Python dependencies
# =============================================================================
step "Installing Python dependencies"
conda run -n "$CONDA_ENV" pip install --quiet -r requirements.txt
# Verify key imports
conda run -n "$CONDA_ENV" python -c "
import fastapi, sqlalchemy, asyncpg, torch, sklearn, fastmcp, aiosqlite
print('  All key imports verified')
"
ok "Python deps installed"

# =============================================================================
# 4. PostgreSQL (local dev) — skip if --docker-only or --skip-pg
# =============================================================================
if ! $DOCKER_ONLY && $WITH_PG; then
  step "PostgreSQL (local dev)"
  PG_DB="netguard"; PG_USER="netguard"; PG_PASS="netguard_dev"; PG_PORT=5432

  if ! command -v psql &>/dev/null; then
    warn "psql not found — installing PostgreSQL 16"
    sudo apt-get update -qq
    sudo apt-get install -y postgresql postgresql-contrib
    sudo service postgresql start
  fi

  if ! pg_isready -q -p "$PG_PORT" 2>/dev/null; then
    warn "PostgreSQL not running — starting"
    sudo service postgresql start
    sleep 2
  fi
  ok "PostgreSQL running on :$PG_PORT"

  # Create user + DB (idempotent, requires sudo)
  sudo -u postgres psql -v ON_ERROR_STOP=0 <<SQL 2>&1 | grep -E "^(NOTICE|ERROR)" || true
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${PG_USER}') THEN
    CREATE USER ${PG_USER} WITH PASSWORD '${PG_PASS}';
    RAISE NOTICE 'Created user ${PG_USER}';
  ELSE
    RAISE NOTICE 'User ${PG_USER} already exists';
  END IF;
END\$\$;
SQL

  sudo -u postgres psql -v ON_ERROR_STOP=0 <<SQL 2>&1 | grep -E "^(NOTICE|ERROR)" || true
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_database WHERE datname='${PG_DB}') THEN
    PERFORM dblink_exec('', 'CREATE DATABASE ${PG_DB} OWNER ${PG_USER}');
    RAISE NOTICE 'Created database ${PG_DB}';
  ELSE
    RAISE NOTICE 'Database ${PG_DB} already exists';
  END IF;
END\$\$;
SQL

  # Simpler fallback
  sudo -u postgres createuser --no-superuser --no-createrole --no-createdb \
    "$PG_USER" 2>/dev/null || true
  sudo -u postgres psql -c "ALTER USER ${PG_USER} WITH PASSWORD '${PG_PASS}';" 2>/dev/null || true
  sudo -u postgres createdb --owner="$PG_USER" "$PG_DB" 2>/dev/null || true

  if PGPASSWORD="$PG_PASS" psql -U "$PG_USER" -h 127.0.0.1 -p "$PG_PORT" \
      -d "$PG_DB" -c '\q' 2>/dev/null; then
    ok "PostgreSQL DB '$PG_DB' accessible"
  else
    warn "Could not verify local PG access. Run manually if needed:"
    echo "    sudo -u postgres psql -c \"CREATE USER netguard WITH PASSWORD 'netguard_dev';\""
    echo "    sudo -u postgres psql -c \"CREATE DATABASE netguard OWNER netguard;\""
    echo "  Or use Docker mode: docker compose up --build"
  fi
fi

# =============================================================================
# 5. Docker images — skip if --local-only
# =============================================================================
if ! $LOCAL_ONLY; then
  step "Docker"
  if ! command -v docker &>/dev/null; then
    warn "docker not found — skipping image build (install Docker Desktop or Docker Engine)"
  else
    ok "docker $(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)"
    if docker compose version &>/dev/null 2>&1; then
      echo "  Building images (this may take a few minutes)..."
      docker compose build --quiet
      ok "Docker images built"
    else
      warn "docker compose not available — skipping image build"
    fi
  fi
fi

# =============================================================================
# 6. .env file
# =============================================================================
step ".env configuration"
if [ -f ".env" ]; then
  ok ".env exists — keeping existing config"
else
  warn "Creating .env from template"
  cat > .env <<'EOF'
# ── App ──────────────────────────────────────────────────────────────────────
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

# ── Database ─────────────────────────────────────────────────────────────────
# Local dev (SQLite — no setup required):
# DATABASE_URL=sqlite+aiosqlite:///./netguard_dev.db
# Local dev (PostgreSQL — run setup.sh to create the user/DB):
DATABASE_URL=postgresql+asyncpg://netguard:netguard_dev@127.0.0.1:5432/netguard

# ── Models ───────────────────────────────────────────────────────────────────
RF_MODEL_PATH=models/rf_cicids.joblib
AE_MODEL_PATH=models/lstm_ae_cicids.pt
SCALER_PATH=models/scaler.joblib

# ── AE threshold (update after training) ─────────────────────────────────────
AE_THRESHOLD=0.008927

# ── Replay / MCP ─────────────────────────────────────────────────────────────
REPLAY_SPEED_MS=200
MCP_PORT=8001
EOF
  ok ".env created"
fi

# =============================================================================
# 7. Required directories
# =============================================================================
step "Runtime directories"
mkdir -p models models/reports data/cicids2017 data/nslkdd logs
ok "models/  data/  logs/  ready"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}  ╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}  ║        Setup complete                  ║${NC}"
echo -e "${GREEN}  ╚════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo "    1. Download datasets into data/cicids2017/ and data/nslkdd/"
echo "    2. Train models:  bash ops.sh train"
echo "    3. Start servers: bash ops.sh start               (local, SQLite)"
echo "                      bash ops.sh start --docker       (Docker, PostgreSQL)"
echo ""
echo "  PostgreSQL local setup (optional, production-only):"
echo "    bash setup.sh --with-pg"
echo ""
