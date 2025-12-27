#!/usr/bin/env bash
# Incoming: .env, environment --- {str, dotenv}
# Processing: authentication, routing, validation --- {3 jobs: authentication, routing, validation}
# Outgoing: .kube/config --- {YAML, text}

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEBUG_LOG_PATH="${DEBUG_LOG_PATH:-$ROOT/.cursor/debug.log}"
DEBUG_RUN_ID="${DEBUG_RUN_ID:-pre-fix}"

debug_log() {
  local hypothesis_id="$1"; shift
  local location="$1"; shift
  local message="$1"; shift
  local data_json="${1:-{}}"
  # IMPORTANT: never log secrets (tokens). data_json must not contain token material.
  mkdir -p "$(dirname "$DEBUG_LOG_PATH")" 2>/dev/null || true

  DEBUG_HYPOTHESIS_ID="$hypothesis_id" \
  DEBUG_LOCATION="$location" \
  DEBUG_MESSAGE="$message" \
  DEBUG_DATA_JSON="$data_json" \
  DEBUG_RUN_ID="$DEBUG_RUN_ID" \
  DEBUG_LOG_PATH="$DEBUG_LOG_PATH" \
  python3 - <<'PY' >/dev/null 2>&1 || true
import json, os, time

path = os.environ.get("DEBUG_LOG_PATH", "")
run_id = os.environ.get("DEBUG_RUN_ID", "run")
hyp = os.environ.get("DEBUG_HYPOTHESIS_ID", "")
loc = os.environ.get("DEBUG_LOCATION", "")
msg = os.environ.get("DEBUG_MESSAGE", "")
raw = os.environ.get("DEBUG_DATA_JSON", "{}")

data = {"raw": raw}

entry = {
    "id": f"log_{int(time.time()*1000)}",
    "timestamp": int(time.time() * 1000),
    "sessionId": "debug-session",
    "runId": run_id,
    "hypothesisId": hyp,
    "location": loc,
    "message": msg,
    "data": data,
}

with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

debug_log "H1" "oc_login.sh:entry" "start" "{\"root\":\"$ROOT\"}"

OPENSHIFT_SERVER_DEFAULT="https://api.os.dcs.gla.ac.uk:6443"
OPENSHIFT_PROJECT_DEFAULT="425krish"

OPENSHIFT_SERVER="${OPENSHIFT_SERVER:-$OPENSHIFT_SERVER_DEFAULT}"
OPENSHIFT_PROJECT="${OPENSHIFT_PROJECT:-$OPENSHIFT_PROJECT_DEFAULT}"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$ROOT/.kube/config}"

resolve_oc_bin() {
  if [[ -n "${OC_BIN:-}" ]]; then
    if [[ -x "$OC_BIN" ]]; then
      printf '%s\n' "$OC_BIN"
      return 0
    fi
    return 2
  fi

  local candidate
  for candidate in "$ROOT/bin/oc" "$ROOT/oc/oc"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v oc >/dev/null 2>&1; then
    command -v oc
    return 0
  fi

  return 1
}

OC_BIN_RESOLVED=""
if OC_BIN_RESOLVED="$(resolve_oc_bin)"; then
  debug_log "H2" "oc_login.sh:resolve_oc_bin" "resolved oc binary" "{\"ocBin\":\"$OC_BIN_RESOLVED\"}"
else
  rc=$?
  debug_log "H2" "oc_login.sh:resolve_oc_bin" "failed to resolve oc binary" "{\"rc\":$rc,\"expected1\":\"$ROOT/bin/oc\",\"expected2\":\"$ROOT/oc/oc\"}"
  if [[ $rc -eq 2 ]]; then
    echo "ERROR: OC_BIN is set but not executable: $OC_BIN" >&2
    exit 1
  fi
  echo "ERROR: oc binary not found. Expected $ROOT/bin/oc or $ROOT/oc/oc, or oc on PATH." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" && -z "${TOKEN:-}" && -z "${OPENSHIFT_TOKEN:-}" ]]; then
  debug_log "H3" "oc_login.sh:token" "no env file and no token env vars" "{\"envFile\":\"$ENV_FILE\"}"
  echo "ERROR: no token source. Create $ENV_FILE with TOKEN=... or export TOKEN/OPENSHIFT_TOKEN" >&2
  exit 1
fi

TOKEN_SOURCE="env_file"
if [[ -n "${TOKEN:-}" ]]; then
  TOKEN_SOURCE="TOKEN_env"
elif [[ -n "${OPENSHIFT_TOKEN:-}" ]]; then
  TOKEN_SOURCE="OPENSHIFT_TOKEN_env"
fi
debug_log "H3" "oc_login.sh:token" "token source selected" "{\"source\":\"$TOKEN_SOURCE\",\"envFile\":\"$ENV_FILE\"}"

TOKEN_RESOLVED=""
if [[ "$TOKEN_SOURCE" == "TOKEN_env" ]]; then
  TOKEN_RESOLVED="$TOKEN"
elif [[ "$TOKEN_SOURCE" == "OPENSHIFT_TOKEN_env" ]]; then
  TOKEN_RESOLVED="$OPENSHIFT_TOKEN"
else
  # Extract TOKEN without printing it. Accept optional surrounding single/double quotes.
  TOKEN_RESOLVED="$(
    grep -E '^TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2-
  )"
  TOKEN_RESOLVED="${TOKEN_RESOLVED#\"}"; TOKEN_RESOLVED="${TOKEN_RESOLVED%\"}"
  TOKEN_RESOLVED="${TOKEN_RESOLVED#\'}"; TOKEN_RESOLVED="${TOKEN_RESOLVED%\'}"
fi

if [[ -z "$TOKEN_RESOLVED" ]]; then
  debug_log "H3" "oc_login.sh:token" "token resolved empty" "{\"source\":\"$TOKEN_SOURCE\"}"
  echo "ERROR: token resolved empty (check $ENV_FILE TOKEN=... or TOKEN/OPENSHIFT_TOKEN env vars)" >&2
  exit 1
fi

umask 077
mkdir -p "$(dirname "$KUBECONFIG_PATH")"
export KUBECONFIG="$KUBECONFIG_PATH"
debug_log "H5" "oc_login.sh:kubeconfig" "kubeconfig path set" "{\"kubeconfig\":\"$KUBECONFIG_PATH\"}"

TLS_ARGS=()
TLS_MODE="default"
if [[ -n "${OPENSHIFT_CA_FILE:-}" ]]; then
  TLS_ARGS+=(--certificate-authority="$OPENSHIFT_CA_FILE")
  TLS_MODE="ca_file"
elif [[ "${OPENSHIFT_SKIP_TLS_VERIFY:-true}" == "true" ]]; then
  TLS_ARGS+=(--insecure-skip-tls-verify=true)
  TLS_MODE="insecure_skip_tls_verify"
fi
debug_log "H4" "oc_login.sh:tls" "tls mode selected" "{\"mode\":\"$TLS_MODE\"}"

set +e
OC_LOGIN_STDERR="$("$OC_BIN_RESOLVED" login --token="$TOKEN_RESOLVED" --server="$OPENSHIFT_SERVER" "${TLS_ARGS[@]}" 2>&1 >/dev/null)"
LOGIN_RC=$?
set -e
if [[ $LOGIN_RC -ne 0 ]]; then
  # Do not include tokens. oc stderr should not contain token material; still truncate defensively.
  OC_LOGIN_STDERR_TRUNC="${OC_LOGIN_STDERR:0:500}"
  debug_log "H4" "oc_login.sh:oc_login" "oc login finished (failed)" "{\"rc\":$LOGIN_RC,\"server\":\"$OPENSHIFT_SERVER\",\"stderr\":\"$OC_LOGIN_STDERR_TRUNC\"}"
else
  debug_log "H4" "oc_login.sh:oc_login" "oc login finished (ok)" "{\"rc\":$LOGIN_RC,\"server\":\"$OPENSHIFT_SERVER\"}"
fi
if [[ $LOGIN_RC -ne 0 ]]; then
  if [[ "$OC_LOGIN_STDERR" == *"token provided is invalid or expired"* ]]; then
    debug_log "H4" "oc_login.sh:status" "classified failure: new token needed" "{\"rc\":$LOGIN_RC}"
    echo "new token needed"
  else
    debug_log "H4" "oc_login.sh:status" "classified failure: other" "{\"rc\":$LOGIN_RC}"
    echo "other things: oc login failed"
    echo "oc error (truncated): ${OC_LOGIN_STDERR:0:500}" >&2
  fi
  exit $LOGIN_RC
fi

set +e
OC_PROJECT_STDERR="$("$OC_BIN_RESOLVED" project "$OPENSHIFT_PROJECT" 2>&1 >/dev/null)"
PROJECT_RC=$?
set -e
if [[ $PROJECT_RC -ne 0 ]]; then
  OC_PROJECT_STDERR_TRUNC="${OC_PROJECT_STDERR:0:500}"
  debug_log "H5" "oc_login.sh:oc_project" "oc project finished (failed)" "{\"rc\":$PROJECT_RC,\"project\":\"$OPENSHIFT_PROJECT\",\"stderr\":\"$OC_PROJECT_STDERR_TRUNC\"}"
else
  debug_log "H5" "oc_login.sh:oc_project" "oc project finished (ok)" "{\"rc\":$PROJECT_RC,\"project\":\"$OPENSHIFT_PROJECT\"}"
fi
if [[ $PROJECT_RC -ne 0 ]]; then
  debug_log "H5" "oc_login.sh:status" "classified failure: other (project)" "{\"rc\":$PROJECT_RC}"
  echo "other things: oc project failed"
  echo "oc error (truncated): ${OC_PROJECT_STDERR:0:500}" >&2
  exit $PROJECT_RC
fi

WHOAMI="$("$OC_BIN_RESOLVED" whoami 2>/dev/null || true)"
SERVER="$("$OC_BIN_RESOLVED" whoami --show-server 2>/dev/null || true)"
PROJ="$("$OC_BIN_RESOLVED" project -q 2>/dev/null || true)"
debug_log "H5" "oc_login.sh:exit" "success" "{\"whoami\":\"$WHOAMI\",\"server\":\"$SERVER\",\"project\":\"$PROJ\"}"

debug_log "H5" "oc_login.sh:status" "classified success: login done" "{}"
echo "login done"
echo "$WHOAMI @ $SERVER :: $PROJ"


