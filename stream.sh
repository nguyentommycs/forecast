#!/usr/bin/env bash
# Starts the full streaming pipeline:
#   1. Docker (Zookeeper + Kafka + Redis)
#   2. Kafka consumer (background, logs to streaming/consumer.log)
#   3. Kafka producer (foreground)
#
# Usage:
#   ./stream.sh                  # default 3600x replay speed
#   ./stream.sh --speed 0        # unlimited speed
#   ./stream.sh --speed 100      # 100x real-time

set -e

VENV="$(dirname "$0")/.venv/Scripts/python"
LOG_FILE="$(dirname "$0")/streaming/consumer.log"
SPEED_ARGS="$*"

# --- 1. Start Docker services ---
echo "[1/3] Starting Docker services..."
docker compose up -d
echo "      Waiting for Kafka to be ready..."

# Poll until Kafka responds (max 60s)
for i in $(seq 1 30); do
  if docker compose exec -T kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
    echo "      Kafka is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Kafka did not become ready in time." >&2
    exit 1
  fi
  sleep 2
done

# --- 2. Start consumer in background ---
echo "[2/3] Starting consumer (logs -> streaming/consumer.log)..."
"$VENV" streaming/consumer.py >"$LOG_FILE" 2>&1 &
CONSUMER_PID=$!
echo "      Consumer PID: $CONSUMER_PID"
sleep 2  # give consumer time to connect

# --- 3. Start producer in foreground ---
echo "[3/3] Starting producer..."
echo "      Press Ctrl+C to stop."
echo ""
"$VENV" streaming/producer.py $SPEED_ARGS

# Clean up consumer when producer exits
echo ""
echo "Producer finished. Stopping consumer (PID $CONSUMER_PID)..."
kill "$CONSUMER_PID" 2>/dev/null || true
echo "Done."
