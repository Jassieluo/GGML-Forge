#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/root/GGML-Forge
DATA_DIR=/root/autodl-tmp/models
LOG_DIR=/root/logs
CONFIG_DIR=/root/.config/ggml-forge
CONFIG_FILE="$CONFIG_DIR/server.env"
CONFIG_EXAMPLE="$PROJECT_DIR/scripts/server.env.example"
PID_FILE="$LOG_DIR/forge-server.pid"
LOG_FILE="$LOG_DIR/forge-server.log"
KEY_FILE="$CONFIG_DIR/api-key"
BINARY="$PROJECT_DIR/build-x64-linux-cuda-release/bin/forge-server"

is_server_pid() {
    local candidate=$1
    [[ "$candidate" =~ ^[0-9]+$ ]] || return 1
    [[ -e "/proc/$candidate/exe" ]] || return 1
    local executable
    executable=$(readlink "/proc/$candidate/exe")
    executable=${executable% (deleted)}
    [[ "$executable" == "$BINARY" ]]
}

mkdir -p "$LOG_DIR" "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "Created default configuration: $CONFIG_FILE"
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${FORGE_HOST:=127.0.0.1}"
: "${FORGE_PORT:=6006}"
: "${FORGE_DEVICE:=CUDA0}"
: "${FORGE_THREADS:=16}"
: "${FORGE_MAX_CONCURRENCY:=1}"
: "${FORGE_LLM_MODEL_DIR:=/root/autodl-tmp/models/Qwen3.6-35B-A3B}"
: "${FORGE_LLM_MODEL_NAME:=Qwen3.6-35B-A3B}"
: "${FORGE_LLM_QUANT:=Q4_K_M}"
: "${FORGE_LLM_CONTEXT:=131072}"
: "${FORGE_LLM_GPU_LAYERS:=-1}"
: "${FORGE_ASR_MODEL:=/root/autodl-tmp/models/asr/whisper_cpp/whisper-small-q4_0.bin}"
: "${FORGE_TTS_MODEL:=/root/autodl-tmp/models/tts/gpt_sovits/configs/v2-q4.json}"
if [[ -z "${FORGE_LLM_MODEL:-}" ]]; then
    FORGE_LLM_MODEL="$FORGE_LLM_MODEL_DIR/$FORGE_LLM_MODEL_NAME-$FORGE_LLM_QUANT.gguf"
fi

for value in FORGE_PORT FORGE_THREADS FORGE_MAX_CONCURRENCY FORGE_LLM_CONTEXT; do
    if [[ ! "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid positive integer in $CONFIG_FILE: $value=${!value}" >&2
        exit 1
    fi
done
if [[ ! -f "$FORGE_LLM_MODEL" ]]; then
    echo "LLM model not found: $FORGE_LLM_MODEL" >&2
    echo "Check FORGE_LLM_QUANT or FORGE_LLM_MODEL in $CONFIG_FILE" >&2
    exit 1
fi
if [[ ! -f "$FORGE_ASR_MODEL" ]]; then
    echo "ASR model not found: $FORGE_ASR_MODEL" >&2
    exit 1
fi
if [[ ! -f "$FORGE_TTS_MODEL" ]]; then
    echo "TTS model not found: $FORGE_TTS_MODEL" >&2
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    pid=$(<"$PID_FILE")
    if is_server_pid "$pid" && kill -0 "$pid" 2>/dev/null; then
        echo "forge-server is already running (PID $pid)"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "forge-server binary not found: $BINARY" >&2
    echo "Build it with: cd $PROJECT_DIR && /root/miniconda3/bin/cmake --build --preset x64-linux-cuda-release" >&2
    exit 1
fi

if [[ ! -s "$KEY_FILE" ]]; then
    umask 077
    openssl rand -hex 32 > "$KEY_FILE"
fi
chmod 600 "$KEY_FILE"
api_key=$(<"$KEY_FILE")

cd "$PROJECT_DIR"
nohup "$BINARY" \
    --host "$FORGE_HOST" \
    --port "$FORGE_PORT" \
    --api-key "$api_key" \
    --device "$FORGE_DEVICE" \
    --threads "$FORGE_THREADS" \
    --max-concurrency "$FORGE_MAX_CONCURRENCY" \
    --llm-context "$FORGE_LLM_CONTEXT" \
    --llm-gpu-layers "$FORGE_LLM_GPU_LAYERS" \
    --llm-model "$FORGE_LLM_MODEL" \
    --asr-model "$FORGE_ASR_MODEL" \
    --tts-model "$FORGE_TTS_MODEL" \
    >> "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "forge-server exited during startup; inspect $LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS \
        -H "Authorization: Bearer $api_key" \
        "http://$FORGE_HOST:$FORGE_PORT/health" >/dev/null 2>&1; then
        # A stale server may still own the port. Give bind failures time to
        # terminate and verify that this exact child remains alive.
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "forge-server exited after health check; another process may own $FORGE_HOST:$FORGE_PORT" >&2
            rm -f "$PID_FILE"
            exit 1
        fi
        echo "forge-server is ready (PID $pid, context $FORGE_LLM_CONTEXT, quant $FORGE_LLM_QUANT)"
        echo "Config: $CONFIG_FILE"
        echo "Log: $LOG_FILE"
        echo "Chat: /root/miniconda3/bin/python $PROJECT_DIR/scripts/forge-chat.py"
        exit 0
    fi
    sleep 1
done

echo "forge-server did not become healthy within 180 seconds; inspect $LOG_FILE" >&2
exit 1
