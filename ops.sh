#!/usr/bin/env bash
# =============================================================================
# NetGuard IDS — Operations Script
# Usage: bash ops.sh <command> [--docker] [service]
#
# Commands:
#   start   [--docker]          Start all services (or Docker stack)
#   stop    [--docker]          Stop all services
#   restart [--docker] [svc]    Restart all or a specific service
#   status  [--docker]          Show status of all services
#   logs    [--docker] [svc]    Tail logs (all or specific service)
#   train   [stage]             Trigger a training run (rf | ae1 | ae2 | all)
#   db      start|stop|status   Manage the PostgreSQL service
#
# Docker services: db  api  ui  mcp
# Local services:  db  api  mcp
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="dl_env"
PID_DIR="$SCRIPT_DIR/logs"
API_PID_FILE="$PID_DIR/api.pid"
MCP_PID_FILE="$PID_DIR/mcp.pid"
API_LOG="$PID_DIR/api.log"
MCP_LOG="$PID_DIR/mcp.log"

mkdir -p "$PID_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  [OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}  [WARN]${NC} $*"; }
fail() { echo -e "${RED}  [FAIL]${NC} $*"; exit 1; }
info() { echo -e "${CYAN}  -->  ${NC}$*"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
COMMAND="${1:-help}"
DOCKER_MODE=false
SERVICE=""
shift || true

for arg in "$@"; do
  case "$arg" in
    --docker) DOCKER_MODE=true ;;
    *)        SERVICE="$arg"   ;;
  esac
done

# =============================================================================
# Docker helpers
# =============================================================================
docker_start() {
  info "Starting Docker stack..."
  docker compose up -d ${SERVICE:+--scale $SERVICE=1}
  sleep 3
  docker compose ps
  ok "Stack started — dashboard: http://localhost  API docs: http://localhost:8000/docs"
}

docker_stop() {
  info "Stopping Docker stack..."
  docker compose down
  ok "Stack stopped"
}

docker_restart() {
  info "Restarting ${SERVICE:-all services}..."
  if [ -n "$SERVICE" ]; then
    docker compose restart "$SERVICE"
  else
    docker compose restart
  fi
  ok "Restart done"
}

docker_status() {
  docker compose ps
}

docker_logs() {
  if [ -n "$SERVICE" ]; then
    docker compose logs -f "$SERVICE"
  else
    docker compose logs -f
  fi
}

# =============================================================================
# Local helpers
# =============================================================================

_is_running() {
  local pidfile="$1"
  [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

local_db_start() {
  if pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null; then
    ok "PostgreSQL already running"
    return
  fi
  info "Starting PostgreSQL..."
  sudo service postgresql start
  sleep 2
  pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null && ok "PostgreSQL started" \
    || warn "PostgreSQL may not have started — check: sudo service postgresql status"
}

local_db_stop() {
  info "Stopping PostgreSQL..."
  sudo service postgresql stop
  ok "PostgreSQL stopped"
}

local_db_status() {
  if pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null; then
    ok "PostgreSQL is running on :5432"
  else
    warn "PostgreSQL is NOT running"
  fi
}

local_api_start() {
  if _is_running "$API_PID_FILE"; then
    ok "API already running (PID $(cat "$API_PID_FILE"))"
    return
  fi
  info "Starting API server on :8000 ..."
  PYTHONPATH="$SCRIPT_DIR" conda run -n "$CONDA_ENV" \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    >> "$API_LOG" 2>&1 &
  echo $! > "$API_PID_FILE"
  sleep 4
  if _is_running "$API_PID_FILE"; then
    ok "API started (PID $(cat "$API_PID_FILE")) — http://localhost:8000/docs"
  else
    fail "API failed to start — check logs: $API_LOG"
  fi
}

local_api_stop() {
  if _is_running "$API_PID_FILE"; then
    info "Stopping API (PID $(cat "$API_PID_FILE"))..."
    kill "$(cat "$API_PID_FILE")" 2>/dev/null || true
    rm -f "$API_PID_FILE"
    ok "API stopped"
  else
    warn "API is not running"
  fi
}

local_mcp_start() {
  if _is_running "$MCP_PID_FILE"; then
    ok "MCP server already running (PID $(cat "$MCP_PID_FILE"))"
    return
  fi
  info "Starting MCP server on :8001 ..."
  PYTHONPATH="$SCRIPT_DIR" conda run -n "$CONDA_ENV" \
    python -m mcp_server.server \
    >> "$MCP_LOG" 2>&1 &
  echo $! > "$MCP_PID_FILE"
  sleep 3
  if _is_running "$MCP_PID_FILE"; then
    ok "MCP started (PID $(cat "$MCP_PID_FILE")) — localhost:8001"
  else
    warn "MCP may not have started — check logs: $MCP_LOG"
  fi
}

local_mcp_stop() {
  if _is_running "$MCP_PID_FILE"; then
    info "Stopping MCP (PID $(cat "$MCP_PID_FILE"))..."
    kill "$(cat "$MCP_PID_FILE")" 2>/dev/null || true
    rm -f "$MCP_PID_FILE"
    ok "MCP stopped"
  else
    warn "MCP is not running"
  fi
}

local_status() {
  echo ""
  echo "  Service      Status           PID     Endpoint"
  echo "  ─────────────────────────────────────────────────────"

  # PostgreSQL
  if pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null; then
    printf "  %-12s ${GREEN}%-16s${NC} %-7s %s\n" "db" "running" "-" ":5432"
  else
    printf "  %-12s ${YELLOW}%-16s${NC} %-7s %s\n" "db" "stopped" "-" ""
  fi

  # API
  if _is_running "$API_PID_FILE"; then
    printf "  %-12s ${GREEN}%-16s${NC} %-7s %s\n" \
      "api" "running" "$(cat "$API_PID_FILE")" "http://localhost:8000"
    # Health check
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
      printf "  %-12s ${GREEN}%-16s${NC}\n" "" "  health OK"
    else
      printf "  %-12s ${YELLOW}%-16s${NC}\n" "" "  health failing"
    fi
  else
    printf "  %-12s ${YELLOW}%-16s${NC} %-7s\n" "api" "stopped" "-"
  fi

  # MCP
  if _is_running "$MCP_PID_FILE"; then
    printf "  %-12s ${GREEN}%-16s${NC} %-7s %s\n" \
      "mcp" "running" "$(cat "$MCP_PID_FILE")" "localhost:8001"
  else
    printf "  %-12s ${YELLOW}%-16s${NC} %-7s\n" "mcp" "stopped" "-"
  fi

  echo ""
}

local_logs() {
  case "${SERVICE:-api}" in
    api) tail -f "$API_LOG" ;;
    mcp) tail -f "$MCP_LOG" ;;
    *)   fail "Unknown service: $SERVICE — use 'api' or 'mcp'" ;;
  esac
}

# =============================================================================
# Train command (triggers API training endpoint or CLI directly)
# =============================================================================
do_train() {
  local stage="${SERVICE:-all}"
  info "Starting training: stage=$stage (source=csv)"
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    # API is running — use training endpoint
    local response
    response=$(curl -s -X POST http://localhost:8000/training/start \
      -H "Content-Type: application/json" \
      -d "{\"stage\": \"$stage\", \"data_source\": \"csv\"}")
    local run_id
    run_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id','?'))")
    ok "Training queued — run_id=$run_id"
    info "Poll status: curl http://localhost:8000/training/status/$run_id"
    info "Or check UI Training tab: http://localhost:8000"
  else
    # API not running — use CLI directly
    warn "API not running — running training CLI directly"
    PYTHONPATH="$SCRIPT_DIR" conda run -n "$CONDA_ENV" \
      python -m training.pipelines.training_pipeline "$stage"
  fi
}

# =============================================================================
# DB sub-command
# =============================================================================
do_db() {
  local sub="${SERVICE:-status}"
  case "$sub" in
    start)  local_db_start  ;;
    stop)   local_db_stop   ;;
    status) local_db_status ;;
    *) fail "Usage: ops.sh db start|stop|status" ;;
  esac
}

# =============================================================================
# Dispatch
# =============================================================================
case "$COMMAND" in
  start)
    if $DOCKER_MODE; then docker_start
    else
      local_db_start
      local_api_start
      local_mcp_start
      local_status
    fi
    ;;

  stop)
    if $DOCKER_MODE; then docker_stop
    else
      local_mcp_stop
      local_api_stop
      # Leave DB running by default (shared resource); stop with: ops.sh db stop
      info "DB left running. Stop with: bash ops.sh db stop"
    fi
    ;;

  restart)
    if $DOCKER_MODE; then docker_restart
    else
      local_api_stop; sleep 1; local_api_start
      local_mcp_stop; sleep 1; local_mcp_start
    fi
    ;;

  status)
    if $DOCKER_MODE; then docker_status
    else local_status
    fi
    ;;

  logs)
    if $DOCKER_MODE; then docker_logs
    else local_logs
    fi
    ;;

  train) do_train ;;

  db) do_db ;;

  help|--help|-h|"")
    echo ""
    echo "  Usage: bash ops.sh <command> [--docker] [service]"
    echo ""
    echo "  Commands:"
    echo "    start   [--docker]         Start all services"
    echo "    stop    [--docker]         Stop all services"
    echo "    restart [--docker] [svc]   Restart (svc = api | mcp | db | ui)"
    echo "    status  [--docker]         Show service status"
    echo "    logs    [--docker] [svc]   Tail logs (default: api)"
    echo "    train   [stage]            Trigger training (rf|ae1|ae2|all)"
    echo "    db      start|stop|status  Manage PostgreSQL"
    echo ""
    echo "  Examples:"
    echo "    bash ops.sh start                  # start locally"
    echo "    bash ops.sh start --docker         # start via Docker compose"
    echo "    bash ops.sh restart api            # restart API only"
    echo "    bash ops.sh logs api               # tail API logs"
    echo "    bash ops.sh train rf               # trigger RF training"
    echo "    bash ops.sh stop --docker          # stop Docker stack"
    echo ""
    ;;

  *)
    fail "Unknown command: $COMMAND — run 'bash ops.sh help'"
    ;;
esac
