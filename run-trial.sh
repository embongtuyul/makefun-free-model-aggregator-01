#!/bin/bash
# Run one local practice trial: start env, run the server, score the result.
# Usage: ./run-trial.sh 0|1|2 [port]
set -u
TRIAL="${1:-0}"
PORT="${2:-18080}"
KIT="$HOME/workspace/makefun/kits/free-model-aggregator-01-kit"
ENTRY="$HOME/workspace/makefun/entries/free-model-aggregator-01/main.py"
STATE="/tmp/fma-state$TRIAL"
SCORES="/tmp/fma-scores$TRIAL"
rm -rf "$STATE" "$SCORES"
mkdir -p "$STATE" "$SCORES"

# Sim provider keys (from practice/trial-N/input/env/providers.json).
export GROQ_API_KEY=sim-groq-0001
export CEREBRAS_API_KEY=sim-cerebras-0002
export GEMINI_API_KEY=sim-gemini-0003
export ZAI_API_KEY=sim-zai-0004
export OPENROUTER_API_KEY=sim-openrouter-0005
export SAMBANOVA_API_KEY=sim-sambanova-0006
export CLOUDFLARE_API_TOKEN=sim-cloudflare-0007

cd "$KIT"
MAKEFUN_STATE_DIR="$STATE" \
MAKEFUN_ENV_DATA_DIR="$KIT/practice/trial-$TRIAL/input/env" \
MAKEFUN_ENV_PORT="$PORT" \
node program/env/server.mjs > "/tmp/fma-env$TRIAL.log" 2>&1 &
ENV_PID=$!
sleep 1

START=$(date +%s)
MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$TRIAL/input/prompt.txt" \
python3 "$ENTRY"
SERVER_RC=$?
END=$(date +%s)
echo "server exit=$SERVER_RC wall=$((END-START))s"

kill -TERM "$ENV_PID" 2>/dev/null
wait "$ENV_PID" 2>/dev/null

if [ ! -f "$STATE/export/state.json" ]; then
  echo "NO STATE EXPORTED"; tail -20 "/tmp/fma-env$TRIAL.log"; exit 1
fi

MAKEFUN_OUTPUT_DIR="$STATE/export" \
MAKEFUN_REFERENCE_DIR="$KIT/practice/trial-$TRIAL/reference" \
MAKEFUN_SCORES_DIR="$SCORES" \
node program/scorer/score.mjs
echo "--- score ---"
cat "$SCORES"/scores.json 2>/dev/null | python3 -m json.tool
