#!/bin/bash
PROJECT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Docker infrastructure..."
docker compose -f "$PROJECT/docker-compose.yml" up -d

source "$PROJECT/.venv/bin/activate"

osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && source .venv/bin/activate && python streaming/consumer.py\""
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && source .venv/bin/activate && python streaming/producer.py\""
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && source .venv/bin/activate && uvicorn api.main:app --port 8000\""
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && source .venv/bin/activate && streamlit run dashboard/app.py\""
