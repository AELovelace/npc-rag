#!/usr/bin/env bash
# One-time Ubuntu 26.04 LTS setup for an npc-rag node. Run from the checkout as root:
#   sudo git clone https://github.com/AELovelace/npc-rag.git /opt/npc-rag
#   sudo bash /opt/npc-rag/linux/build-llama.sh      # llama-server builds (HIP + Vulkan) into /opt/llama.cpp
#   sudo bash /opt/npc-rag/linux/install-ubuntu.sh     # this script
# Port layout: 9090 left free (Cockpit, if you ever add it), 9091 the npc-rag control page, 9092 the npc-rag API,
# 9100+ the student llama-server copies (127.0.0.1 only, never opened in the firewall).
# Safe to run again: it keeps an existing .env and only adds what is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # The checkout this script lives in.
SERVICE_USER="${SERVICE_USER:-npcrag}"                     # Both services run as this user.
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"                   # Ubuntu 26.04 ships 3.14; the Decider stack is tested on 3.12, so uv fetches it.
UV_HOME="${UV_HOME:-/opt/uv}"                              # uv itself and its Python builds, readable by the service user.
WITH_DECIDER="${WITH_DECIDER:-1}"                          # 1 = install the Decider (the classifier: 9091 is no longer a model port).

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }
echo "npc-rag checkout: $ROOT"

# 1. Packages: git, a C++ toolchain for llama-cpp-python (Decider), the system Python (only to bootstrap uv), ufw.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git build-essential cmake ninja-build python3 python3-venv ufw vulkan-tools
apt-get install -y rocm-smi rocminfo || echo "rocm-smi/rocminfo not in the archive: the control page shows GPUs once ROCm is installed."

# 2. Service user with GPU access (render/video) and journal access for the control page.
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --home-dir "$ROOT" --shell /sbin/nologin "$SERVICE_USER"
fi
usermod -aG render,video,systemd-journal "$SERVICE_USER"

# 3. uv (from PyPI, in its own small venv) installs Python $PYTHON_VERSION under $UV_HOME, then the checkout's .venv.
if [[ ! -x "$UV_HOME/bootstrap/bin/uv" ]]; then
  python3 -m venv "$UV_HOME/bootstrap"
  "$UV_HOME/bootstrap/bin/pip" install --upgrade uv
fi
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"   # Not root's home: the service user must be able to run this Python.
"$UV_HOME/bootstrap/bin/uv" python install "$PYTHON_VERSION"
chmod -R a+rX "$UV_HOME"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV_HOME/bootstrap/bin/uv" venv --seed --python "$PYTHON_VERSION" "$ROOT/.venv"   # --seed: pip inside, so later updates need no uv.
fi
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
if [[ "$WITH_DECIDER" == "1" ]]; then
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements-decider.txt"   # CPU llama-cpp-python build: takes a few minutes.
fi

# 4. .env: start from .env.example the first time, then make sure the node layout keys are set.
ENV="$ROOT/.env"
NEW_PASSWORD=""
FRESH=""
if [[ ! -f "$ENV" ]]; then
  cp "$ROOT/.env.example" "$ENV"
  FRESH=1   # A brand-new file takes every node default below; an existing one keeps its own choices.
fi
set_env() {  # set_env KEY VALUE [missing]: replace KEY=... in place, or append it. "missing" keeps a value an older .env already has.
  local only_missing="${3:-}"
  [[ -n "$FRESH" ]] && only_missing=""
  "$ROOT/.venv/bin/python" - "$ENV" "$1" "$2" "$only_missing" <<'PY'
import re, sys
path, key, value, only_missing = sys.argv[1:5]
lines = open(path, encoding="utf-8").read().splitlines()
pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
hits = [i for i, line in enumerate(lines) if pattern.match(line)]
if hits and only_missing and lines[hits[0]].split("=", 1)[1].strip():
    sys.exit(0)  # Already has a value: keep it.
if hits:
    lines[hits[0]] = f"{key}={value}"
else:
    lines.append(f"{key}={value}")
open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
}
set_env STUDENT_FLEET on missing
set_env STUDENT_HOST 127.0.0.1
set_env STUDENT_BASE_PORT 9100
set_env STUDENT_RESERVED_PORTS 9090,9091
set_env NPC_HOST 0.0.0.0 missing
set_env NPC_PORT 9092
set_env ADMIN_HOST 0.0.0.0 missing
set_env ADMIN_PORT 9091
set_env ADMIN_SERVICE npc-rag.service
if [[ "$WITH_DECIDER" == "1" ]]; then
  set_env CLASSIFIER_BACKEND decider   # The old llamacpp classifier lived on 9091, which is the control page now.
fi
if ! grep -qE '^\s*ADMIN_PASSWORD=.+' "$ENV"; then
  NEW_PASSWORD="$("$ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(18))')"
  set_env ADMIN_PASSWORD "$NEW_PASSWORD"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$ROOT"
chmod 600 "$ENV"   # Keys live here: only the service user reads it.

# 5. systemd units (paths rewritten to this checkout) and the polkit rule that lets the page restart npc-rag.
for unit in npc-rag.service npc-rag-admin.service npc-rag-update.service; do
  sed -e "s#/opt/npc-rag#$ROOT#g" -e "s#^User=npcrag#User=$SERVICE_USER#" -e "s#^Group=npcrag#Group=$SERVICE_USER#" \
    "$ROOT/linux/$unit" > "/etc/systemd/system/$unit"
done
sed -e "s#\"npcrag\"#\"$SERVICE_USER\"#" "$ROOT/linux/50-npc-rag-admin.rules" > /etc/polkit-1/rules.d/50-npc-rag-admin.rules
chmod 644 /etc/polkit-1/rules.d/50-npc-rag-admin.rules
systemctl daemon-reload
systemctl enable --now npc-rag-admin.service
systemctl enable npc-rag.service
systemctl restart npc-rag.service

# 6. Firewall (ufw): the control page and the API only. The 9100+ copies stay on 127.0.0.1.
ufw allow 9091/tcp comment 'npc-rag control page'
ufw allow 9092/tcp comment 'npc-rag API'
if ! ufw status | grep -q "Status: active"; then
  echo "ufw is off, so nothing is blocked yet. To turn it on without locking yourself out: ufw allow OpenSSH && ufw enable"
fi

echo
echo "npc-rag is starting (first start downloads the student GGUF; watch: journalctl -u npc-rag -f)."
echo "Control page: http://$(hostname -I | awk '{print $1}'):9091"
if [[ -n "$NEW_PASSWORD" ]]; then
  echo "Control page password (saved as ADMIN_PASSWORD in $ENV): $NEW_PASSWORD"
fi
if ! grep -qE '^\s*NPC_API_KEY=.+' "$ENV"; then
  echo "NOTE: NPC_API_KEY is empty. Set it on the control page (Secrets) to the game server's NPC_RAG_KEY, then restart."
fi
