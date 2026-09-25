# npc-rag: the LiDollQuest tutorial NPC

A CPU-only agent that answers players through a tutorial NPC (default name **Pip**). A player's message is:

1. **Classified** (port **9091**): is it a question about the game, or general conversation?
2. **Retrieved** (port **9092**): game questions get the best-matching sections of the public player wiki.
3. **Generated** (port **9090**): the llama.cpp server writes Pip's reply from her persona plus those wiki sections.

```
game server ──POST /v1/npc/chat──▶ 9091 classifier + agent ──▶ 9092 wiki RAG
                                         │
                                         └──▶ 9090 llama.cpp (47.51.162.110) ──▶ reply ──▶ game server
```

Nothing here needs a GPU. Embeddings use `BAAI/bge-small-en-v1.5` through fastembed (ONNX runtime, about 7 ms per message on CPU). llama.cpp is the only large model, and it already runs on the AI server.

## Install on the AI server (Windows)

```powershell
cd C:\Scripts\npc-rag
powershell -ExecutionPolicy Bypass -File ps\Install-NpcRag.ps1   # venv, packages, .env, model download, wiki index, tests
notepad .env                                                     # set NPC_API_KEY; check LLM_URL
powershell -ExecutionPolicy Bypass -File ps\Start-NpcRag.ps1     # starts 9092 then 9091, waits for both
powershell -ExecutionPolicy Bypass -File ps\Test-NpcRag.ps1 -ApiKey <key>
```

To run at boot and restart after crashes, open an **Administrator** PowerShell and run:

```powershell
powershell -ExecutionPolicy Bypass -File ps\Register-NpcRagTasks.ps1 -AllowFrom <game server IP>
```

This registers the tasks `npc-rag rag` and `npc-rag agent` as SYSTEM. `-AllowFrom` opens port 9091 to that address only. Port 9092 stays on localhost. `-Remove` undoes both.

**LLM_URL:** the default is `http://47.51.162.110:9090`, as requested. If npc-rag runs on the same machine as llama.cpp, set `LLM_URL=http://127.0.0.1:9090`. Many routers don't let a machine reach its own public IP. From the development PC the public address timed out, but `http://192.168.1.188:9090` worked.

Logs go to `logs\rag.log` and `logs\agent.log`. Every chat request logs its player id, category, method, confidence, time and the first 80 characters of the message.

## The API (port 9091)

Send `Authorization: Bearer <NPC_API_KEY>` or `X-Api-Key: <NPC_API_KEY>`. Without a key configured the service is open, so only run it that way on a private network.

### `POST /v1/npc/chat`

```json
{ "message": "how do I change my diaper?", "player_id": "char-123", "player_name": "Doll" }
```

| Field | Notes |
|---|---|
| `message` | Required, 1 to 500 characters (`MAX_MESSAGE_CHARS`). |
| `player_id` | Optional. Gives the player a short rolling memory (6 messages, forgotten after 30 minutes of silence) so follow-ups like "and in Arcadia?" work. |
| `player_name` | Optional. Pip addresses the player by it. |
| `npc_name` | Optional. Overrides `NPC_NAME` for this call, so one service can voice several guides. |
| `history` | Optional list of `{role: "player"/"npc", content}`. When sent, it replaces the built-in memory (for a game server that stores conversations itself). |

Response:

```json
{
  "reply": "Open your inventory, select a clean diaper, and equip it in the underwear slot. ...",
  "npc_name": "Pip",
  "category": "game",
  "confidence": 1.0,
  "method": "knn",
  "relevant": true,
  "fallback": false,
  "sources": [{"title": "Needs & recovery", "section": "Wearing and changing protection",
               "url": "https://lidoll.dev/wiki/#/needs-and-care/wearing-and-changing-protection", "score": 0.75}],
  "timings_ms": {"classify": 5, "retrieve": 36, "generate": 6606, "total": 6648}
}
```

- `reply` is one plain-text paragraph of at most 600 characters (`MAX_REPLY_CHARS`), with no Markdown, so it can go straight into the dialogue box.
- `fallback: true` means llama.cpp was down or too slow. Pip then says a canned line, pointing at the best wiki section when there is one. Players always get an answer.
- `sources` link into the public wiki. They are empty for small talk, and for game questions the wiki doesn't really cover.
- **429** means that player already has a reply on the way. **422** means the message is empty or too long. **401** means a missing or wrong key.

Other endpoints:
- `POST /v1/classify {message}`: returns the classification only.
- `DELETE /v1/npc/session/{player_id}`: forgets a player's memory.
- `GET /health`: reports whether llama.cpp and the RAG service are reachable. It needs no key.

### Port 9092 (RAG, localhost only)

- `POST /v1/search {query, k}`: returns the fused top-k sections, `best_dense` and `relevant`.
- `POST /v1/reindex {source?}`: rebuilds from the wiki without a restart. If the rebuild fails, the old index keeps serving.
- `GET /health`: returns the index metadata.

## How each step works

**Classifier** (`npc_rag/classifier.py`). This runs on CPU, cheapest step first, and stops at the first confident answer:
1. **kNN.** The message is compared with the labelled examples in `data/classifier_seeds.json`, using the mean of the 3 closest game examples against the 3 closest general ones. A gap of at least `CLASSIFIER_MARGIN` (0.04) decides it. This settles about 90% of messages in about 5 ms.
2. **Wiki evidence.** If kNN is unsure, the RAG service is asked how strongly the wiki covers the message. A score of `RAG_GAME_EVIDENCE` (0.66) or more makes it a game question.
3. **LLM tiebreak.** If it's still unsure, llama.cpp answers with one word, GAME or CHAT (about 1 s). Turn this off with `CLASSIFIER_LLM_TIEBREAK=0`.

Accuracy on the 80 held-out messages in `data/eval_messages.json` is 95.0% with CPU only and **98.8%** with the tiebreak. Measure it with `python -m npc_rag eval [--llm]`. The one known miss is "who are the gods": it sits right next to "who are you", and the wiki says "patron", not "god". Pip answers it as small talk, says she isn't sure, and points to the wiki instead of inventing gods.

**Retrieval** (`npc_rag/wiki.py`, `npc_rag/index.py`). Chapters come from `pages.json` and `content/<slug>.md` (the public list, so maintainer-only pages are never indexed). They are cut at every heading. Tables become one sentence per row, and long sections split at about 180 words with one block of overlap. That gives 73 chunks for the current wiki. Each search blends BM25 keywords (exact words like "Cursebreaker" or "RPP") and embeddings (meaning) with reciprocal rank fusion. Heading ids replicate `wiki.js`, and a test runs the real regex in node, so every link opens the right section.

**Generation** (`npc_rag/prompts.py`). The system prompt holds the rules, `data/persona.md` and the wiki excerpts. Pip must answer game questions from the excerpts only, and say she's not sure (and suggest a wiki page) when they don't cover it. In small talk she never states game facts. Qwen3's thinking is switched off per request (`enable_thinking: false`), and any `<think>` text is stripped anyway. Replies measured against the live server: about 3 s for small talk, about 7 s for a game answer.

## Everyday tasks

| Task | How |
|---|---|
| The wiki changed | `python -m npc_rag build-index`, then restart 9092, or `POST /v1/reindex` with no restart. |
| Pip misclassifies a real message | Add it to the right list in `data/classifier_seeds.json`, restart 9091, run `python -m npc_rag eval --llm`. Never add eval messages to the seeds. |
| Change Pip's personality | Edit `data/persona.md` and restart 9091. The rules in `prompts.py` stay in force. |
| Try a question | `python -m npc_rag search "pay toilets"` (retrieval only) or `python -m npc_rag ask "where do I sleep?"` (the full pipeline, against the running agent). |
| Use a local wiki copy | Set `WIKI_SOURCE=C:\path\to\lidollquest` (or its `web\wiki` folder). |
| Run the tests | `.venv\Scripts\python -m pytest`. They use a fixture wiki, a hashing embedder and a mocked llama.cpp, so they need no network. |

All settings are listed with comments in `.env.example`.

## Not done yet: the in-game NPC

This service is the brain. The game side still needs:
1. A Lidollquest-server command that forwards a player's message to `POST /v1/npc/chat` with the API key. It should use the character id as `player_id`, rate-limit per player, and answer asynchronously, because replies take 3 to 8 s.
2. A GML text-entry dialogue for talking to Pip, plus an NPC placed in the starting hubs.
3. GM panel and game editor toggles: an on/off switch and the NPC's name.
