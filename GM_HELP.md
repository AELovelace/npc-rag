# GM handbook assistant (port 9093)

This second CPU RAG service answers online GM authoring questions from the **GM wiki**. The player assistant on 9092, its index, and the classifier on 9091 are unchanged. Both assistants can use the loaded llama.cpp model on 9090. No extra model download is needed if the existing embedding cache is populated.

## Network path and access

Admin browser -> authenticated game server `/gm/help/chat` -> AI server `:9093/v1/gm/chat` -> llama.cpp `:9090`.

The game server forwards the answer and canonical wiki references to the browser. **Only the game server makes requests to RAG**: keep your public-IP allowlist and allow the game server's public outbound address on port 9093. If the game server uses a different outbound address, that address must be allowed; browser users' IPs are irrelevant. No CORS opening, public browser access, or port forwarding to every admin is needed. Images load from the game server's `/gm/wiki/assets/` route.

The game server revalidates the LiDollID gamemaster role on every question and before delivering an answer. A separate service key protects port 9093 even within the allowed network. The browser never receives this key, endpoint, or model credentials. Prompts contain only the question, six recent chat messages, and retrieved documentation; no drafts, wallets, flags or accounts are attached. There are no execution/editing tools.

## Configure the AI server

Use the existing virtual environment (`ps\Install-NpcRag.ps1` if this is a new installation). Add these entries to the existing `.env`; preserve existing NPC settings:

```dotenv
GM_HOST=0.0.0.0
GM_PORT=9093
GM_API_KEY=replace-with-a-long-random-service-key
GM_WIKI_SOURCE=C:/path/to/Lidollquest-server/server/gm-wiki
GM_INDEX_DIR=data/gm-index
GM_LLM_URL=http://127.0.0.1:9090
GM_MAX_TOKENS=1100
GM_LLM_TIMEOUT=60
GM_MIN_RELEVANCE=0.5
```

Choose a fresh random key and put the same value in the game server's `LIDOLLQUEST_GM_HELP_KEY`. Do not use an admin's login token. The service refuses to start without its key; its default bind address is loopback until you explicitly set `GM_HOST`.

`GM_WIKI_SOURCE` must contain `pages.json` and `content/*.md`. Copy the game's entire `server/gm-wiki` directory to the AI server or use a readable HTTPS wiki root such as `https://YOUR-GAME-SERVER/gm/wiki/`. That URL must be accessible from the AI server under the game server's existing GM address/TLS rules. Using a local copy avoids that inbound network dependency. Do not point this at the player wiki.

```powershell
.\.venv\Scripts\python.exe -m npc_rag gm-build-index
.\ps\Start-GmRag.ps1
```

For startup/restart management, from Administrator PowerShell:

```powershell
.\ps\Register-GmRagTask.ps1 -AllowFrom YOUR_GAME_SERVER_PUBLIC_IP
```

This creates only the `npc-rag GM help` task and its separate restricted firewall rule. It never changes the player NPC task. For a custom GM_PORT, also pass `-Port` with the same number. Inspect Task Scheduler for service errors, or run Start-GmRag.ps1 in a console to see startup diagnostics.

## Configure the game server

```dotenv
LIDOLLQUEST_GM_HELP_URL=http://YOUR_AI_SERVER_ADDRESS:9093
LIDOLLQUEST_GM_HELP_KEY=the-same-random-service-key
```

Use the address reachable **from the game server** (LAN, VPN or public routing as appropriate). Keep the existing public-IP restriction. Restart the game server after configuring these variables. The editor and Advanced GM panel offer **Ask the GM wiki assistant**, opening `/gm/help` in a small window; a normal link still works when pop-ups are blocked.

## Updating the handbook

The GM index is a snapshot. After wiki changes, sync the GM Markdown to the AI server, stop only the GM help task, run `gm-build-index`, then restart it. Build success prints page/chunk counts and a timestamp. Stop before rebuilding so readers cannot observe a partially replaced multi-file index. The player index is separate and must not be reused as GM_INDEX_DIR.

Screenshot captions are embedded alongside section text. The service returns matching section images as structured references, and the game server verifies them against its shipped chapters and illustration manifest. The UI lets admins expand each source, enlarge screenshots, and open the original section. This is text retrieval with documented illustrations, **not image understanding**. Keep Markdown image alt text descriptive; links must use `../assets/tutorial/<name>.png` or `.svg`, with the image in `illustrations.json`. Keep AI and game-server wiki copies in sync.

## Operation and troubleshooting

- `GET http://127.0.0.1:9093/health` reports `service: gm-rag`, the page count and index timestamp; it does not test model health.
- A missing key or unreadable wiki fails startup clearly. Confirm that the SYSTEM task can read the wiki folder and embedding cache.
- “Not configured” in the editor means the game server is missing its help URL or key.
- An unavailable-service error suggests a listener, address, firewall, key or timeout problem. Test connectivity **from the game server**, not from an admin phone.
- If llama.cpp fails after retrieval, the UI still displays the matching wiki sections and pictures. If evidence is too weak, it asks for a more specific question rather than inventing instructions.
- Each process allows two concurrent questions; the game proxy allows one in flight per staff account. Excess requests receive a retryable busy response; game-state transactions are never held open while waiting for AI.
- Chat is kept only in tab memory; reload, New chat, account changes, or 30 minutes of inactivity clear it. Request contents are not logged by this service. Model-server logging is controlled separately.
- Tests: `.\.venv\Scripts\python.exe -m pytest`. GM tests use a deterministic CPU embedder and fake model, without downloading anything or contacting production.
