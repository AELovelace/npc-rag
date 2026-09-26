# npc-rag: the LiDollQuest tutorial NPC

Wiki retrieval and code-level reply guards for the tutorial NPC (default name **Pip**). The models themselves run in Sakura (DoLLAMACPP-Frontend) on the AI server. npc-rag decides what they see and checks what they say.

A player's message goes through four steps:

1. **Classify** (Sakura, port **9091**): a small model answers GAME or CHAT.
2. **Retrieve** (npc-rag, port **9092**): game questions get the best-matching sections of the public player wiki.
3. **Generate** (Sakura, port **9090**): the main model writes Pip's reply from her persona plus those sections.
4. **Guard** (npc-rag): replies that invent names or recite the rules are retried, then replaced.

```
game server --POST /v1/npc/chat--> 9092 npc-rag --GAME or CHAT?--> 9091 Sakura classifier model
                                       |   wiki index + guard
                                       +--persona + wiki sections--> 9090 Sakura main model --> reply --> game server
```

npc-rag needs no GPU. Embeddings use `BAAI/bge-small-en-v1.5` through fastembed (ONNX runtime, about 7 ms per message on CPU).

## Install on the AI server (Windows)

Start both Sakura model slots first (9090 and 9091), then:

```powershell
cd C:\Scripts\npc-rag
powershell -ExecutionPolicy Bypass -File ps\Install-NpcRag.ps1   # venv, packages, .env, model download, wiki index, tests
notepad .env                                                     # set NPC_API_KEY
powershell -ExecutionPolicy Bypass -File ps\Start-NpcRag.ps1     # starts 9092 and checks it can reach 9090 and 9091
powershell -ExecutionPolicy Bypass -File ps\Test-NpcRag.ps1 -ApiKey <key>
```

To run at boot and restart after crashes, open an **Administrator** PowerShell and run:

```powershell
powershell -ExecutionPolicy Bypass -File ps\Register-NpcRagTasks.ps1 -AllowFrom <game server IP>
```

This registers the task `npc-rag` as SYSTEM, starting 30 s after boot so Sakura's models come up first. `-AllowFrom` opens port 9092 to that address only. `-Remove` undoes both. It also removes the `npc-rag rag` and `npc-rag agent` tasks and the port 9091 firewall rule from the earlier two-service layout.

`LLM_URL` and `CLASSIFIER_URL` default to `127.0.0.1:9090` and `127.0.0.1:9091`, which is right when npc-rag runs on the same machine as Sakura. Many routers don't let a machine reach its own public IP.

Logs go to `logs\npc-rag.log`. Every chat request logs its player id, category, how it was classified, the guard's verdict, the time taken and the first 80 characters of the message.

## The API (port 9092)

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
  "method": "llm",
  "relevant": true,
  "fallback": false,
  "guard": "passed",
  "sources": [{"title": "Needs & recovery", "section": "Wearing and changing protection",
               "url": "https://lidoll.dev/wiki/#/needs-and-care/wearing-and-changing-protection", "score": 0.75}],
  "timings_ms": {"classify": 610, "retrieve": 36, "generate": 3100, "total": 3750}
}
```

- `reply` is one plain-ASCII paragraph of at most 600 characters (`MAX_REPLY_CHARS`), with no Markdown, so it can go straight into the dialogue box.
- `method` is `llm` when the classifier model decided, or `wiki` when it was down and the wiki score decided.
- `guard` says what happened to the reply (see [The guard](#the-guard)).
- `fallback: true` means the main model was down or too slow. Pip then says a canned line, pointing at the best wiki section when there is one. Players always get an answer.
- `sources` link the wiki sections the reply was written from. They are empty for small talk and whenever the guard replaced the reply.
- **429** means that player already has a reply on the way. **422** means the message is empty or too long. **401** means a missing or wrong key.

Other endpoints:
- `POST /v1/search {query, k}`: returns the fused top-k sections, `best_dense` and `relevant`.
- `POST /v1/reindex {source?}`: rebuilds from the wiki without a restart. If the rebuild fails, the old index keeps serving.
- `POST /v1/classify {message}`: returns the classification only.
- `DELETE /v1/npc/session/{player_id}`: forgets a player's memory.
- `GET /health`: reports the index and whether both model servers are reachable. It needs no key.

## How each step works

**Classifier** (`npc_rag/classifier.py`). The model on `CLASSIFIER_URL` gets a short instruction and answers GAME or CHAT (4 tokens, temperature 0). From the development PC, Llama-3.2-3B on MONOLITH-III classified 96.2% of the 80 held-out messages in `data/eval_messages.json` correctly, taking about 0.6 s each. If the model is down, times out (`CLASSIFIER_TIMEOUT`) or answers something else, a best wiki score of `RAG_GAME_EVIDENCE` (0.66) or more makes the message a game question, and anything lower counts as chat. Measure accuracy with `python -m npc_rag eval`, or `eval --wiki-only` for the fallback alone.

**Retrieval** (`npc_rag/wiki.py`, `npc_rag/index.py`). Chapters come from `pages.json` and `content/<slug>.md` (the public list, so maintainer-only pages are never indexed). They are cut at every heading. Tables become one sentence per row, and long sections split at about 180 words with one block of overlap. That gives 73 chunks for the current wiki. Each search blends BM25 keywords (exact words like "Cursebreaker" or "RPP") and embeddings (meaning) with reciprocal rank fusion. Heading ids replicate `wiki.js`, and a test runs the real regex in node, so every link opens the right section.

**Generation** (`npc_rag/prompts.py`). The system prompt holds the rules, `data/persona.md` and the wiki sections the guard kept. Pip must answer game questions from those sections only, and in small talk she never states game facts. Qwen3's thinking is switched off per request (`enable_thinking: false`), and any `<think>` text is stripped anyway.

### The guard

Smaller models like Josiefied-Qwen3-8B ignore "only use the notes" often enough that prompt wording isn't sufficient. In testing, the 8B invented "Dragon's Hollow" and a Dragon Trainer, named gods the wiki never mentions, answered a diaper question from the Utopia section, and recited its instructions when asked. `npc_rag/guard.py` checks for these in code:

| Check | What happens | `guard` value |
|---|---|---|
| A game question with no section scoring at least `MIN_RELEVANCE` and at least `GUARD_MIN_WORDS` (15) words long | The model is never called; Pip says she isn't sure and names the closest wiki page. | `no_relevant_notes` |
| Sections scoring more than `GUARD_KEEP_MARGIN` below the best one | Dropped before the prompt, so a weak runner-up can't steer the answer. | |
| The reply names something (a capitalised word mid-sentence) that the notes, rules, persona, conversation and `GUARD_ALLOWED_TERMS` never mention | One retry at temperature 0.1 that names the words to avoid (`GUARD_RETRIES`), then the "not sure" line. | `passed_after_retry` or `unsupported_terms` |
| The reply shares three or more 6-word runs with the rules or persona | Replaced with a stock "that's between me and my makers" line. | `leak_blocked` |
| Curly quotes, dashes, ellipses, accents | Flattened to plain ASCII, because GameMaker fonts often lack them. | |

Otherwise `guard` is `passed`, or `llm_unavailable` when the main model couldn't be reached. One known gap: an invented name that only ever appears as the first word of a sentence isn't caught.

## Everyday tasks

| Task | How |
|---|---|
| The wiki changed | `python -m npc_rag build-index`, then restart npc-rag, or `POST /v1/reindex` with no restart. |
| Pip refuses a real name the wiki lacks | Add it to `GUARD_ALLOWED_TERMS` in `.env` (or better, to the wiki), then restart npc-rag. |
| Pip misclassifies a message | Adjust `CLASSIFY_PROMPT` in `npc_rag/classifier.py`, restart, and run `python -m npc_rag eval`. |
| Change Pip's personality | Edit `data/persona.md` and restart npc-rag. The rules in `prompts.py` stay in force. |
| Try a question | `python -m npc_rag search "pay toilets"` (retrieval only) or `python -m npc_rag ask "where do I sleep?"` (the full pipeline, against the running service). |
| Use a local wiki copy | Set `WIKI_SOURCE=C:\path\to\lidollquest` (or its `web\wiki` folder). |
| Run the tests | `.venv\Scripts\python -m pytest`. They use a fixture wiki, a hashing embedder and mocked model servers, so they need no network. |

All settings are listed with comments in `.env.example`.

## Not done yet: the in-game NPC

This service is the brain. The game side still needs:
1. A Lidollquest-server command that forwards a player's message to `POST /v1/npc/chat` on port 9092 with the API key. It should use the character id as `player_id`, rate-limit per player, and answer asynchronously, because replies take several seconds.
2. A GML text-entry dialogue for talking to Pip, plus an NPC placed in the starting hubs.
3. GM panel and game editor toggles: an on/off switch and the NPC's name.
