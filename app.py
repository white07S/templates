#!/usr/bin/env bash
# Install PostgreSQL from a local source tarball and build pgvector
# Usage:
#   ./install_postgres_from_tar.sh \
#     --tar /path/to/postgresql-16.2.tar.gz \
#     --prefix "$HOME/postgresql" \
#     --data "$HOME/postgresql_data" \
#     --port 5432 \
#     [--pgvector-tag v0.8.0]
#
# Notes:
# - Expects a *source* tarball like postgresql-16.2.tar.gz (not a prebuilt binary)
# - Builds and installs pgvector using PGXS with your new pg_config
# - Does not start the server (use the second script)

set -euo pipefail

# Defaults
TARBALL=""
PREFIX="${HOME}/postgresql"
DATA_DIR="${HOME}/postgresql_data"
PORT="5432"
PGVECTOR_TAG="v0.8.0"   # latest widely-available as of late 2024–2025

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar)           TARBALL="$2"; shift 2;;
    --prefix)        PREFIX="$2"; shift 2;;
    --data|--data-dir) DATA_DIR="$2"; shift 2;;
    --port)          PORT="$2"; shift 2;;
    --pgvector-tag)  PGVECTOR_TAG="$2"; shift 2;;
    -h|--help)
      grep -E '^# (Usage|Notes):' "$0" | sed 's/^# //'; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${TARBALL}" ]]; then
  echo "ERROR: --tar /path/to/postgresql-XX.Y.tar.gz is required" >&2
  exit 1
fi
if [[ ! -f "${TARBALL}" ]]; then
  echo "ERROR: tarball not found: ${TARBALL}" >&2
  exit 1
fi

echo "==> Installing PostgreSQL from: ${TARBALL}"
echo "    Prefix: ${PREFIX}"
echo "    Data:   ${DATA_DIR}"
echo "    Port:   ${PORT}"
echo "    pgvector tag: ${PGVECTOR_TAG}"

# Basic build sanity checks
command -v gcc >/dev/null || { echo "ERROR: gcc not found. Install build tools (e.g., build-essential)"; exit 1; }
command -v make >/dev/null || { echo "ERROR: make not found."; exit 1; }
command -v git  >/dev/null || { echo "ERROR: git not found."; exit 1; }

# Create dirs
mkdir -p "${PREFIX}" "${DATA_DIR}"

# Extract sources to a temp build dir
BUILD_ROOT="$(mktemp -d /tmp/pgsrc.XXXXXX)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT
tar -xzf "${TARBALL}" -C "${BUILD_ROOT}"
SRCDIR="$(find "${BUILD_ROOT}" -maxdepth 1 -type d -name "postgresql-*.*" | head -n1)"
if [[ -z "${SRCDIR}" ]]; then
  echo "ERROR: could not find extracted source dir" >&2
  exit 1
fi

# Build & install PostgreSQL (per official docs)
pushd "${SRCDIR}" >/dev/null
./configure --prefix="${PREFIX}"
make -j"$(nproc)"
make install
popd >/dev/null

# Add contrib (optional but useful)
if [[ -d "${SRCDIR}/contrib" ]]; then
  pushd "${SRCDIR}/contrib" >/dev/null
  make -j"$(nproc)"
  make install
  popd >/dev/null
fi

# Environment file
SETUP_ENV="${PREFIX}/setup_env.sh"
cat > "${SETUP_ENV}" <<EOF
#!/usr/bin/env bash
export PATH="${PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="${PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export PGDATA="${DATA_DIR}"
export PGPORT=${PORT}
export PGHOST=localhost
echo "Environment set: PGDATA=\$PGDATA PGPORT=\$PGPORT PATH includes ${PREFIX}/bin"
EOF
chmod +x "${SETUP_ENV}"

# Initialize cluster if empty
if [[ -z "$(ls -A "${DATA_DIR}" 2>/dev/null || true)" ]]; then
  echo "==> Initializing database cluster at ${DATA_DIR}"
  "${PREFIX}/bin/initdb" -D "${DATA_DIR}" --encoding=UTF8 --locale=C
fi

# Minimal config
PG_CONF="${DATA_DIR}/postgresql.conf"
if ! grep -q "^port\s*=\s*${PORT}" "${PG_CONF}"; then
  cat >> "${PG_CONF}" <<EOF

# Custom Configuration
listen_addresses = 'localhost'
port = ${PORT}
max_connections = 100
shared_buffers = 128MB
EOF
fi

# Trust local connections for easy dev usage
cat > "${DATA_DIR}/pg_hba.conf" <<'EOF'
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

# Build & install pgvector against the new pg_config (per README)
echo "==> Building pgvector (${PGVECTOR_TAG})"
export PG_CONFIG="${PREFIX}/bin/pg_config"
pushd /tmp >/dev/null
rm -rf pgvector
git clone --branch "${PGVECTOR_TAG}" https://github.com/pgvector/pgvector.git
cd pgvector
make -j"$(nproc)"
make install
popd >/div/null 2>/dev/null || popd >/dev/null # tolerate subshells

echo "==> Done."
echo "Next steps:"
echo "  1) source ${SETUP_ENV}"
echo "  2) Use ./start_and_list_tables.sh to start and verify (creates extension in DB)"



#!/usr/bin/env bash
# Start PostgreSQL, ensure pgvector is enabled in the target DB, and print table names.
# Usage:
#   ./start_and_list_tables.sh [--env /path/to/setup_env.sh] [--db testdb] [--user $USER]
#
# Behavior:
# - Starts the server with pg_ctl
# - Waits until ready
# - Creates the DB if it doesn't exist
# - Ensures CREATE EXTENSION vector in that DB
# - Prints non-system table names (schema.table)

set -euo pipefail

ENV_FILE=""
DBNAME="testdb"
DBUSER="${USER:-$(id -un)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_FILE="$2"; shift 2;;
    --db)  DBNAME="$2"; shift 2;;
    --user) DBUSER="$2"; shift 2;;
    -h|--help)
      grep -E '^# (Usage|Behavior):' "$0" | sed 's/^# //'; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  endac
done

# Load environment
if [[ -n "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
elif [[ -f "${HOME}/postgresql/setup_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/postgresql/setup_env.sh"
else
  echo "ERROR: Could not find setup_env.sh. Pass --env /path/to/setup_env.sh" >&2
  exit 1
fi

BIN="${PATH%%:*}"  # first path is usually ${PREFIX}/bin due to setup_env
if [[ ! -x "${BIN}/pg_ctl" ]]; then
  echo "ERROR: pg_ctl not found in PATH. Did you source setup_env.sh?" >&2
  exit 1
fi

# Start server if not running
if ! "${BIN}/pg_ctl" -D "${PGDATA}" status >/dev/null 2>&1; then
  echo "==> Starting PostgreSQL..."
  "${BIN}/pg_ctl" -D "${PGDATA}" -l "${PGDATA}/logfile" start
fi

# Wait until ready
echo "==> Waiting for server to be ready on ${PGHOST:-localhost}:${PGPORT:-5432} ..."
for _ in {1..30}; do
  if "${BIN}/pg_isready" >/dev/null 2>&1; then break; fi
  sleep 1
done
"${BIN}/pg_isready"

# Create DB if missing
if ! "${BIN}/psql" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DBNAME}'" | grep -q 1; then
  echo "==> Creating database: ${DBNAME}"
  "${BIN}/createdb" -O "${DBUSER}" "${DBNAME}"
fi

# Ensure pgvector extension in the DB (per README: CREATE EXTENSION vector;)
echo "==> Ensuring pgvector extension exists in ${DBNAME}"
"${BIN}/psql" -d "${DBNAME}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Print non-system tables (schema.table)
echo "==> Listing tables in ${DBNAME} (excluding system schemas)"
SQL="SELECT schemaname || '.' || tablename
     FROM pg_tables
     WHERE schemaname NOT IN ('pg_catalog','information_schema')
     ORDER BY 1;"
"${BIN}/psql" -d "${DBNAME}" -At -c "${SQL}" || true
