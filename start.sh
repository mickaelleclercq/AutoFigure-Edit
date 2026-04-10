#!/bin/bash
cd "$(dirname "$0")"

# Kill existing instances of server.py in this folder
PIDS=$(pgrep -f "python.*$(pwd)/server.py\|python.*AutoFigure-Edit/server.py" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "[start.sh] Stopping existing instances (PIDs: $PIDS)..."
    kill $PIDS 2>/dev/null
    sleep 1
    # Force kill if still alive
    PIDS=$(pgrep -f "python.*$(pwd)/server.py\|python.*AutoFigure-Edit/server.py" 2>/dev/null)
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null && sleep 1
fi

# Free port 8000 if occupied
PORT_PID=$(lsof -ti :8000 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "[start.sh] Port 8000 occupied (PID: $PORT_PID), killing..."
    kill -9 $PORT_PID 2>/dev/null
    # Wait up to 5s for port to be released
    for i in $(seq 1 5); do
        sleep 1
        lsof -ti :8000 >/dev/null 2>&1 || break
        echo "[start.sh] Waiting for port 8000 to be released... ($i/5)"
    done
fi

# Absolute path to the project venv python (never relative, never depends on PATH)
VENV_PYTHON="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"
export AUTOFIGURE_PYTHON="$VENV_PYTHON"
export PYTHONNOUSERSITE=1
# Strip any inherited venv from PATH - only keep system + .venv/bin
export VIRTUAL_ENV="$(cd "$(dirname "$0")" && pwd)/.venv"
export PATH="$VIRTUAL_ENV/bin:$(echo "$PATH" | tr ':' '\n' | grep -vE '(/\.venv|/.venv|/venvs|sim_env)' | tr '\n' ':' | sed 's/:$//')"
unset PYTHONPATH CONDA_PREFIX PYTHONSTARTUP

echo "[start.sh] Starting AutoFigure-Edit on http://localhost:8000"
echo "[start.sh] Python: $VENV_PYTHON"
echo "[start.sh] VIRTUAL_ENV: $VIRTUAL_ENV"
echo "[start.sh] PYTHONNOUSERSITE: $PYTHONNOUSERSITE"

# Verify: print where torch will come from
echo "[start.sh] torch location check:"
PYTHONWARNINGS=ignore "$VENV_PYTHON" -c "
import sys, torch, timm
print('  python:', sys.executable)
print('  torch:', torch.__file__, 'version:', torch.__version__)
print('  torch.cuda:', torch.cuda.__file__)
print('  timm:', timm.__file__)
sim = [p for p in sys.path if 'sim_env' in p]
print('  sim_env in sys.path:', sim if sim else 'CLEAN')
" 2>/dev/null

exec "$VENV_PYTHON" server.py
