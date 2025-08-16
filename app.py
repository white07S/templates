#!/bin/bash

# PostgreSQL Binary Installation Script
# This script downloads PostgreSQL binaries, installs them, and sets up a data directory

set -e  # Exit on error

# Configuration Variables
POSTGRES_VERSION="16.2"
INSTALL_DIR="$HOME/postgresql"
DATA_DIR="$HOME/postgresql_data"
PORT=5432
OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Determine the correct binary package
if [ "$OS_TYPE" = "linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        POSTGRES_PACKAGE="postgresql-${POSTGRES_VERSION}-1-linux-x64-binaries.tar.gz"
        DOWNLOAD_URL="https://get.enterprisedb.com/postgresql/${POSTGRES_PACKAGE}"
    else
        echo "Architecture $ARCH not supported in this script"
        exit 1
    fi
elif [ "$OS_TYPE" = "darwin" ]; then
    POSTGRES_PACKAGE="postgresql-${POSTGRES_VERSION}-1-osx-binaries.zip"
    DOWNLOAD_URL="https://get.enterprisedb.com/postgresql/${POSTGRES_PACKAGE}"
else
    echo "OS $OS_TYPE not supported"
    exit 1
fi

echo "========================================="
echo "PostgreSQL Binary Installation Script"
echo "========================================="
echo "Version: $POSTGRES_VERSION"
echo "Install Directory: $INSTALL_DIR"
echo "Data Directory: $DATA_DIR"
echo "Port: $PORT"
echo "========================================="

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$DATA_DIR"

# Download PostgreSQL binaries
echo "Downloading PostgreSQL $POSTGRES_VERSION..."
cd /tmp
if [ ! -f "$POSTGRES_PACKAGE" ]; then
    curl -L -o "$POSTGRES_PACKAGE" "$DOWNLOAD_URL" || wget -O "$POSTGRES_PACKAGE" "$DOWNLOAD_URL"
else
    echo "Package already downloaded, using existing file..."
fi

# Extract binaries
echo "Extracting PostgreSQL binaries..."
if [[ "$POSTGRES_PACKAGE" == *.tar.gz ]]; then
    tar -xzf "$POSTGRES_PACKAGE" -C "$INSTALL_DIR" --strip-components=1
elif [[ "$POSTGRES_PACKAGE" == *.zip ]]; then
    unzip -q "$POSTGRES_PACKAGE" -d "$INSTALL_DIR"
    mv "$INSTALL_DIR"/pgsql/* "$INSTALL_DIR"
    rmdir "$INSTALL_DIR/pgsql"
fi

# Set up environment variables
echo "Setting up environment variables..."
export PATH="$INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
export PGDATA="$DATA_DIR"

# Initialize the database cluster
echo "Initializing database cluster..."
"$INSTALL_DIR/bin/initdb" -D "$DATA_DIR" --encoding=UTF8 --locale=C

# Configure PostgreSQL
echo "Configuring PostgreSQL..."
cat >> "$DATA_DIR/postgresql.conf" << EOF

# Custom Configuration
listen_addresses = 'localhost'
port = $PORT
max_connections = 100
shared_buffers = 128MB
EOF

# Configure authentication
echo "Configuring authentication..."
cat > "$DATA_DIR/pg_hba.conf" << EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

# Start PostgreSQL
echo "Starting PostgreSQL server..."
"$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" -l "$DATA_DIR/logfile" start

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Create a test database and user
echo "Creating test database and user..."
"$INSTALL_DIR/bin/createdb" -p $PORT testdb
"$INSTALL_DIR/bin/psql" -p $PORT -d postgres -c "CREATE USER testuser WITH PASSWORD 'testpass';"
"$INSTALL_DIR/bin/psql" -p $PORT -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE testdb TO testuser;"

# Install pgvector extension (if needed)
echo "Attempting to install pgvector..."
cd /tmp
git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git 2>/dev/null || true
if [ -d "pgvector" ]; then
    cd pgvector
    export PG_CONFIG="$INSTALL_DIR/bin/pg_config"
    make clean
    make
    make install
    "$INSTALL_DIR/bin/psql" -p $PORT -d testdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
    echo "pgvector extension installed successfully"
else
    echo "Could not clone pgvector repository. You may need to install it manually."
fi

# Create start/stop scripts
echo "Creating management scripts..."

# Start script
cat > "$INSTALL_DIR/start_postgres.sh" << EOF
#!/bin/bash
export PATH="$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PGDATA="$DATA_DIR"
"$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" -l "$DATA_DIR/logfile" start
EOF
chmod +x "$INSTALL_DIR/start_postgres.sh"

# Stop script
cat > "$INSTALL_DIR/stop_postgres.sh" << EOF
#!/bin/bash
export PATH="$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PGDATA="$DATA_DIR"
"$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" stop
EOF
chmod +x "$INSTALL_DIR/stop_postgres.sh"

# Status script
cat > "$INSTALL_DIR/status_postgres.sh" << EOF
#!/bin/bash
export PATH="$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PGDATA="$DATA_DIR"
"$INSTALL_DIR/bin/pg_ctl" -D "$DATA_DIR" status
EOF
chmod +x "$INSTALL_DIR/status_postgres.sh"

# Create environment setup script
cat > "$INSTALL_DIR/setup_env.sh" << EOF
#!/bin/bash
export PATH="$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PGDATA="$DATA_DIR"
export PGPORT=$PORT
export PGHOST=localhost
echo "PostgreSQL environment variables set."
echo "PATH includes: $INSTALL_DIR/bin"
echo "PGDATA: $DATA_DIR"
echo "PGPORT: $PORT"
EOF
chmod +x "$INSTALL_DIR/setup_env.sh"

echo "========================================="
echo "PostgreSQL Installation Complete!"
echo "========================================="
echo "Installation directory: $INSTALL_DIR"
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"
echo ""
echo "Management scripts created:"
echo "  Start:  $INSTALL_DIR/start_postgres.sh"
echo "  Stop:   $INSTALL_DIR/stop_postgres.sh"
echo "  Status: $INSTALL_DIR/status_postgres.sh"
echo "  Environment: source $INSTALL_DIR/setup_env.sh"
echo ""
echo "To use PostgreSQL commands, run:"
echo "  source $INSTALL_DIR/setup_env.sh"
echo ""
echo "Test connection:"
echo "  psql -p $PORT -d testdb -U testuser"
echo "========================================="
