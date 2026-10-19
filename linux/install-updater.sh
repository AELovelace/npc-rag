#!/usr/bin/env bash
# Enable repository updates on an already-installed node, without reinstalling Python/GPU packages.
# Usage: sudo bash /opt/npc-rag/linux/install-updater.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo."; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-$(systemctl show npc-rag-admin.service -p User --value)}"
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || { echo "Cannot determine the existing control-page service user."; exit 1; }
[[ -x "$ROOT/.venv/bin/python" && -f "$ROOT/python/update_repo.py" ]] || { echo "Install the updated repository and its venv first."; exit 1; }
runuser -u "$SERVICE_USER" -- test -w "$ROOT/.git" || { echo "The service user must own a writable Git checkout."; exit 1; }
sed -e "s#/opt/npc-rag#$ROOT#g" -e "s#^User=npcrag#User=$SERVICE_USER#" -e "s#^Group=npcrag#Group=$SERVICE_USER#" \
  "$ROOT/linux/npc-rag-update.service" > /etc/systemd/system/npc-rag-update.service
sed -e "s#\"npcrag\"#\"$SERVICE_USER\"#" "$ROOT/linux/50-npc-rag-admin.rules" > /etc/polkit-1/rules.d/50-npc-rag-admin.rules
chmod 644 /etc/systemd/system/npc-rag-update.service /etc/polkit-1/rules.d/50-npc-rag-admin.rules
systemctl daemon-reload
systemctl restart npc-rag-admin.service # Reload the button/API; the RAG service and its loaded models keep running.
echo "Repository updater installed. Sign in again on port 9091; use Check & install updates."
