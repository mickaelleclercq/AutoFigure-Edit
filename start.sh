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

echo "[start.sh] Starting AutoFigure-Edit on http://localhost:8000"
exec /home/mickael/.venvs/sim_env/bin/python server.py
