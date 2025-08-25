#!/bin/bash
# WiFi-3D-Fusion JavaScript Visualizer Launcher
# This script launches the advanced WiFi-3D-Fusion system with auto-recovery

# Parse command line arguments
USE_DEVICE=""
DEVICE_TYPE=""
INTERFACE="mon0"
PORT=5000

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      USE_DEVICE="true"
      DEVICE_TYPE="$2"
      shift 2
      ;;
    --interface)
      INTERFACE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_wifi3d_js.sh [--device TYPE] [--interface IFACE] [--port PORT]"
      echo "  --device TYPE    : Use real device (supported: monitor, nexmon, esp32)"
      echo "  --interface IFACE: WiFi interface to use (default: mon0)"
      echo "  --port PORT      : HTTP server port (default: 5000)"
      exit 1
      ;;
  esac
done

# Activate Python environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated Python virtual environment"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Activated Python virtual environment"
fi

# Kill any existing instances and free up port
pkill -f run_js_visualizer.py
sleep 1  # Give processes time to terminate

# Check if port is still in use and kill process if needed
PORT_PID=$(lsof -t -i:$PORT 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "🔥 Port $PORT still in use, forcefully freeing it..."
    kill -9 $PORT_PID 2>/dev/null
    sleep 1
fi

echo "🧹 Cleaned up any existing processes"

# Create necessary directories
mkdir -p env/visualization
mkdir -p env/visualization/js
mkdir -p env/visualization/css

# Run the visualizer with auto-recovery
echo "🚀 Starting WiFi-3D-Fusion JavaScript Visualizer..."
echo "🔄 System will auto-recover from freezes"
echo "🌐 Once started, open http://localhost:${PORT}/ in your browser"

# Set up command based on device type
BASE_CMD="python3 run_js_visualizer.py --port ${PORT}"

if [ "$USE_DEVICE" = "true" ]; then
    case $DEVICE_TYPE in
        "monitor")
            echo "📡 Using real WiFi adapter in monitor mode: ${INTERFACE}"
            BASE_CMD="${BASE_CMD} --source monitor --interface ${INTERFACE}"
            ;;
        "nexmon")
            echo "📡 Using Nexmon CSI collection: ${INTERFACE}"
            BASE_CMD="${BASE_CMD} --source nexmon --interface ${INTERFACE}"
            ;;
        "esp32")
            echo "📡 Using ESP32 WiFi device"
            BASE_CMD="${BASE_CMD} --source esp32"
            ;;
        *)
            echo "⚠️ Unknown device type: ${DEVICE_TYPE}, falling back to dummy data"
            BASE_CMD="${BASE_CMD} --dummy"
            ;;
    esac
else
    # No device specified, use dummy data
    BASE_CMD="${BASE_CMD} --dummy"
fi

# Function to check if port is in use
is_port_in_use() {
    local port=$1
    # Try to see if the port is already bound
    if command -v lsof >/dev/null 2>&1; then
        lsof -i :$port >/dev/null 2>&1
        return $?
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep -q ":$port "
        return $?
    else
        # If we can't check, assume it's free
        return 1
    fi
}

# Try to find an available port
MAX_PORT_ATTEMPTS=10
attempt=0
original_port=$PORT

while [ $attempt -lt $MAX_PORT_ATTEMPTS ]; do
    if is_port_in_use $PORT; then
        echo "⚠️ Port $PORT is already in use, trying next port..."
        PORT=$((PORT + 1))
        attempt=$((attempt + 1))
    else
        break
    fi
done

if [ $PORT -ne $original_port ]; then
    echo "🔄 Using alternative port $PORT instead of $original_port"
    BASE_CMD="python3 run_js_visualizer.py --port ${PORT} $(echo $BASE_CMD | cut -d ' ' -f 4-)"
fi

# Run with the constructed command
CMD="$BASE_CMD $@"
echo "🔄 Running command: $CMD"
$CMD

# If we get here, the program was interrupted
echo "👋 WiFi-3D-Fusion JavaScript Visualizer stopped"
