# npc-rag: the LiDollQuest tutorial NPC

## Optional local Decider 2B Q4_K_M classifier (2026-10-14)

Live node B check (October 14, 11:35-11:38 Pacific): `192.168.1.188:9092` reports Decider 2B
Q4_K_M active, four CPU threads, zero GPU layers and one inference slot. The deployed hybrid
96-message suite scored 81/96: GAME 33/48, CHAT 48/48, failing the per-class gate. All 57 completed
model decisions were CHAT; all 33 GAME decisions came from wiki overrides (six social shortcuts).
This does not qualify the model's decision behavior despite its healthy native runtime.

A diagnostic 100%-chat burst sent 100 simultaneous requests to **this node alone**: 100 real
guard-passed replies, median 23.843 s, p95 53.102 s, max 55.063 s, eight under five seconds.
Methods were 10 social, 50 model, 40 wiki fallbacks at approximately the classifier deadline.
Classification median was 7.573 s; generation median was 17.332 s (phase medians do not add).
The ten repeated prompts shared 81 pending requests, with nine fresh model misses and zero cache
hits. This is not diverse-message throughput or a matched comparison with earlier two-node bursts.
Both local and cloud generation saturated (8/4 observed), without cloud cooldown; the node recovered
healthy/idle with zero native inference failures. No production settings changed during the tests.
Artifacts under `C:/Scripts/Lidollquest-server/artifacts/`: `decider-node-b-accuracy-20261014T183516Z.json`
and `burst-20261014183708027.json`. The burst is diagnostic despite the failed accuracy gate;
neither correct chat fallback nor successful HTTP responses qualify gameplay routing for release.

`CLASSIFIER_BACKEND=decider` loads `Mapika/decider-2b-GGUF` / `decider-2b-v11-Q4_K_M.gguf`
inside the existing RAG process. No extra port or Sakura classifier is needed for this backend.
Reply models, wiki retrieval, companions, social shortcuts, label caching and overflow retain
their existing paths. The default remains `llamacpp`, so updating the checkout alone does not switch models.

Decider is an optional Python 3.11+ dependency. Install it into the **RAG virtual environment**:

```powershell
powershell -ExecutionPolicy Bypass -File ps/Install-Decider.ps1
# For a complete initial RAG install, the existing installer also accepts -Decider:
# powershell -ExecutionPolicy Bypass -File ps/Install-NpcRag.ps1 -Decider
```

The upstream package includes torch/transformers even when using GGUF. `llama-cpp-python>=0.3.35`
is needed for this model architecture; a source build on Windows needs Visual Studio C++ Build Tools
and CMake. A working Sakura executable does not install these Python dependencies. Native installation
and live model accuracy/latency must be checked on each host; the unit suite uses a model double.

On Windows, the helper temporarily sets TEMP/TMP/TMPDIR to a short drive-root directory (`C:\t`
for a checkout on C:) and restores their previous values afterward. The earlier checkout-local `.t`
workaround was insufficient: llama-cpp-python 0.3.36 contains an even deeper 178-character archive
member, giving a 254-character pip extraction filename with `C:\t`. The release archive was checked
against PyPI's SHA-256 and all member lengths inspected; native installation still needs a server retry.
The requirements now pin that native package to 0.3.36, within Decider's supported >=0.3.35,<0.4 range.
The helper rejects temp roots longer than eight characters. It does not change the registry or remove
existing temp files. To retry with either the old or new helper immediately:

```powershell
powershell -ExecutionPolicy Bypass -File ps/Install-Decider.ps1 -BuildTemp C:\t
```

The selected directory must be writable. This addresses source extraction; subsequent compilation
still needs the toolchain above. An older incompatible prebuilt wheel is not a substitute.

If CMake reports `nmake` missing and `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` unset, extraction
has succeeded but MSVC is missing or its developer environment is not active. Install **Visual Studio
2022 Build Tools**, selecting **Desktop development with C++**, including the MSVC x64/x86 tools
and a Windows SDK. On Windows LTSC without winget, download and run Microsoft's
[standalone Build Tools 2022 installer](https://aka.ms/vs/17/release/vs_BuildTools.exe).
Select **Desktop development with C++** and keep its recommended components, then click Install.
If Build Tools is already installed without C++, open Visual Studio Installer and choose Modify
to add that workload. The full Visual Studio IDE is not required.

The optional winget equivalent, on machines where winget is available, is:

```powershell
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

After installation, open **Developer PowerShell for VS 2022**, change to the npc-rag checkout, and
run `powershell -ExecutionPolicy Bypass -File ps/Install-Decider.ps1 -BuildTemp C:\t` again.
The updated helper also discovers an installed x64 C++ toolchain using `vswhere` and loads its
developer shell automatically. When native compilation is needed, it checks for `cl.exe` and
`nmake.exe` before pip downloads packages. CMake is already supplied by the isolated wheel build;
installing CMake alone does not supply a C++ compiler. This helper targets Windows x64 builds.
See Microsoft's [developer shell instructions](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022).

The built-in model defaults are:

```dotenv
CLASSIFIER_BACKEND=decider
DECIDER_MODEL=Mapika/decider-2b-GGUF
DECIDER_GGUF_FILE=decider-2b-v11-Q4_K_M.gguf
DECIDER_THREADS=4
DECIDER_GPU_LAYERS=0
DECIDER_CONTEXT=4096
```

`DECIDER_MODEL` also accepts a local directory containing that GGUF, the matching Hugging Face
tokenizer files, and `decider_config.json`. A GGUF alone is insufficient. Relative directories
resolve against this checkout. With a Hub ID, the library downloads required files into the usual
Hugging Face cache on first load; configure `HF_HOME` before launching if a different cache is needed.
First download/load can exceed the launcher's 120-second health wait, so prepare using the evaluation
command before changing the live backend. Startup fails clearly if dependencies/config/model cannot load.

Evaluate without changing the live `.env` or generating any dialogue:

```powershell
.venv\Scripts\python -m npc_rag eval --classifier-backend decider --raw-model --include-mixed --output artifacts/decider-raw.json
.venv\Scripts\python -m npc_rag eval --classifier-backend decider --include-mixed --output artifacts/decider-hybrid.json
```

The second command uses the local wiki index; `--evidence-url http://RAG_HOST:9092` instead uses a
deployed index with the configured NPC_API_KEY. Evaluation records the backend, model, quantization,
decision policy, per-case timings and per-class accuracy. Keep the held-out messages unchanged;
require the existing 90%-per-class gate before a live burst. Model load time is outside per-case timings.
Evaluation loads its own model instance, so avoid running it alongside production Decider on a busy host.

After validation, add the settings above to each RAG host's `.env` and restart the existing 9092 service.
`/health.classification.backend` should be `decider`, with `decision_policy=game-chat-v1`, model/file,
CPU threads, configured GPU layers, context, capacity, busy state and failure count. `classifier_url`
is null because no HTTP classifier is used. Existing `method=llm` and `scores.llm` retain their meanings
as a model-based category, including Decider, for client compatibility. Revert to
`CLASSIFIER_BACKEND=llamacpp` and restart to use the previous CLASSIFIER_URL again.

One worker owns one native context per RAG process. `DECIDER_THREADS` controls CPU threads (1-64),
not parallel model copies. `DECIDER_GPU_LAYERS=0` selects CPU; -1 requests all layers on a GPU with
a compatible llama-cpp-python build. GPU layers accept -1 through 256; context accepts 2048-32768.
Multiple service workers duplicate model memory. Do not enable multi-sequence GGUF batching here:
the upstream model card reports that it can change Q4 probabilities. Queue cancellation prevents
unused requests reaching the native model. An inference already running cannot be stopped by Python
timeout: it keeps its slot until completion, then the next request can proceed. The deadline bounds
the caller's wait, not native CPU runtime. Invalid/failed/overlong-input decisions use existing wiki
fallback and never enter the label cache. Restart RAG after replacing local model/config files.

The adapter uses the model's calibrated option scores, without chat templates, generated labels or
the llama.cpp grammar. Schema validity is not proof of correct routing; no Decider accuracy or
five-second conversation claim is established by the local contract tests.
Validation: 143 local tests pass, including native cancellation/recovery and both evaluation modes.
The dependency-resolution dry run stalled at package metadata and was stopped; no native packages
or model weights were installed by this preparation. Run the install/import check on each RAG host.
Upstream: [model files and scoring instructions](https://huggingface.co/Mapika/decider-2b-GGUF),
[Decider runtime](https://github.com/Mapika/decider).

## Social routing, constrained classification and label caching (2026-10-14)

The `hybrid-v1` policy implements five changes:

1. Whole-message greetings, thanks, farewells and simple check-ins return `general` with
   `method=social`, without calling the classifier or embedding the message. A greeting with an
   extra question (for example, "Hello, how do I equip this?") still takes the normal path.
2. Other messages use a focused gameplay-versus-conversation instruction, balanced examples and
   JSON-quoted input. Strong wiki evidence still overrides a model CHAT decision.
3. The classifier request includes a llama.cpp grammar allowing only `GAME` or `CHAT`.
   The parser rejects extra prose if a backend ignores the constraint. Generation requests do not
   receive this grammar. Format validity alone does not establish classification accuracy.
4. `eval` reports both error directions and accuracy for each class. Its default gate requires
   at least 90% accuracy in each class; a failed gate exits with status 1. The original 80-message
   dataset is unchanged, and `--include-mixed` adds 16 separate mixed/social examples.
5. Each RAG process caches valid model labels for identical trimmed messages, preserving case and
   internal whitespace. Simultaneous identical calls share one model request. Cancelling one
   caller preserves other waiters; cancelling the last releases the upstream slot. Wiki searches,
   persona, history, memories and replies are not cached by this feature.

`CLASSIFIER_CACHE_SIZE=1024` and `CLASSIFIER_CACHE_TTL=300` bound cache size and age in seconds.
Zero size or TTL disables caching and shared calls. Size accepts 0-16384; TTL accepts 0-86400.
Timeouts, invalid labels and errors are not cached. The advertised `/v1/models` identity is checked
at most every 30 seconds when needed: a changed identity invalidates labels; unavailable identity
discovery disables reuse. Restart RAG after a model replacement using the same alias, or after
changing the prompt/configuration, to clear labels immediately. Cache contents never leave the process.
`/health.classification` exposes the policy version and cache counts, without message text.
`CLASSIFIER_TIMEOUT` still includes identity lookup, queueing and waiting on shared model work.

Run the accuracy gate before a burst, using either the local index or a deployed RAG index:

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m npc_rag eval --include-mixed --output artifacts/classifier-eval.json
.venv\Scripts\python -m npc_rag eval --classifier-url http://MODEL_HOST:9094 --evidence-url http://RAG_HOST:9092 --include-mixed --output artifacts/classifier-remote.json
```

`--raw-model` evaluates the model without wiki overrides or social shortcuts. `--wiki-only` skips
model calls. The remote search uses the existing NPC_API_KEY; neither mode changes live settings
or generates paid dialogue. Keep evaluation examples out of prompt tuning. Repeated-prompt burst
results benefit from caching; also measure diverse, previously unseen messages before estimating capacity.

The later ElectricLMA abliterated 1B replacement also failed this policy's gate (58/96 and 52/96).
An updated/old-node burst exposed 45 wrong no-notes responses on the updated node; the old node's
50 chat replies all used wiki-score fallback because unconstrained model responses were invalid.
Do not treat valid constrained labels, cached results or successful fallback replies as proof that
a model follows the classification task. This model/setup is not qualified by the current checks.
The server burst tool accepts `--require-policy hybrid-v1` to reject mixed deployments before load.

Validation on October 14: 122 local tests cover shortcut boundaries, constrained payloads, TTL/LRU,
100-way shared calls, cancellation and model changes. Against the live indexes, the new policy
scored 76/96 and 59/96 with Vikhr 1B Q4 (failed), versus 92/96 and 93/96 with Qwen3-4B Q8 (passed).
These are classification checks, not proof of a five-second complete reply. The user is preparing
an English-oriented abliterated 1B replacement and a dedicated classifier GPU setup; neither is
assumed deployed. Point CLASSIFIER_URL at the chosen classifier after validation and restart the
existing 9092 RAG service. No new RAG listener or game-server URL is needed.

## Classification latency fix (2026-10-14)

Wiki retrieval and the classifier run concurrently. Strong wiki evidence at or above
`RAG_GAME_EVIDENCE` now returns GAME immediately, retaining the retrieved notes and cancelling
pending classifier work. Weak evidence still waits for the model; a later strong match still
overrides CHAT. `scores.llm` is null when the model did not finish before the wiki decided.

`CLASSIFIER_TIMEOUT` now bounds the classifier's slot queue plus HTTP request together. Previously,
queued calls could wait through several timeout windows before their own HTTP timeout began.
The default remains 10 seconds. This does not bound retrieval, generation, or complete request
latency. Cancellation releases the RAG client's slot; backend compute reclamation depends on
the model server handling the disconnected request.

Update this checkout on each RAG host and restart its existing 9092 process using
`powershell -ExecutionPolicy Bypass -File ps/Start-NpcRag.ps1`. No environment change is required.
The 99-test suite includes 100 concurrent classification calls against a stalled mock backend,
slot recovery after cancellation, early wiki decisions, and caller/error cleanup. These are
regression tests, not a live latency measurement. Repeat the live 100-message burst after deployment.

For a CPU/GPU comparison, run the same classifier model and quantization on the same host, first
on CPU and then on the selected GPU in Sakura, keeping its existing port (currently 9094 on the
live fleet), slot count, context, RAG settings and test prompts fixed. Warm each setup equally
and compare full request median/p95, classification timings and guard outcomes. Sharing a GPU
with a reply model may change generation latency too. No GPU configuration is changed by this fix.

## Optional GPT-6 Luna overflow on 9092 (2026-10-14)

Each existing RAG process can detect its own local generation capacity. It retrieves the wiki,
selects allowed lore/memories and assembles the character prompt first, then uses a free local
slot or sends that prepared prompt to GPT-6 Luna. The existing reply guards run afterward.
No second service, additional listening port, game-server setting or firewall opening is needed.

Add these settings to the existing .env on each RAG server that should use overflow:

    OPENAI_API_KEY=<your private API key>
    OPENAI_MODEL=gpt-6-luna
    OPENAI_LOCAL_CAPACITY=4
    OPENAI_CONCURRENCY=4
    OPENAI_TIMEOUT=12

Keep the existing NPC_API_KEY, LLM_URL, LLM_CONCURRENCY and NPC_PORT=9092 settings. Restart
normally with powershell -ExecutionPolicy Bypass -File ps/Start-NpcRag.ps1. A blank or absent
OPENAI_API_KEY disables overflow. A shared NPC_API_KEY is required before paid routing can start.
The OpenAI key belongs only in the RAG environment, never in the game server or client.

OPENAI_LOCAL_CAPACITY is the total number of simultaneous local generations through this RAG
process before it uses overflow. It is capped at LLM_CONCURRENCY times the number of model
copies listed in LLM_URL. Background impressions and summaries count toward local occupancy,
but never use OpenAI themselves; classifier and GM-assistant calls also stay local.
OPENAI_CONCURRENCY limits simultaneous cloud generations per RAG process (not across the fleet).
Capacity settings accept integers 1-64; OPENAI_TIMEOUT accepts positive seconds up to 120.
Requests return to local generation immediately when local slots become available. A fast local
backend failure can also use cloud. These are process slot counts, not GPU utilization or slots
occupied by other applications; adjust the threshold if the model servers are shared.

When cloud capacity is full or cooling down, requests wait locally within LLM_TIMEOUT. A failed
cloud call returns through the existing game failure path and starts a 30-second cloud cooldown.
Queueing, generation and local failover share the caller deadline; cloud calls additionally use
OPENAI_TIMEOUT. There is no promise that all replies finish within five seconds.
Health checks include both backends: /health reports llm=true if either is reachable and exposes
an overflow object with enabled, local/cloud capacity, in-flight counts and cooldown seconds.
Checking OpenAI model access uses GET /v1/models/gpt-6-luna, without generating a paid reply.

The adapter uses the Responses API, reasoning.effort=none, bounded output and store=false.
Prepared dialogue prompts, relevant lore, history and selected memories go to OpenAI.
store=false disables stored Responses; it does not promise zero provider retention.
There are no automatic HTTP retries; the existing content guard may make its corrective retry.
Refusals remain refusals. Concurrency limits are not a monetary spending cap.

The earlier separate .env.overflow/9095 setup is superseded. Use your existing .env instead.
Any old private .env.overflow file is left untouched and Git-ignored so a pasted key is preserved;
it is not read during normal startup. NPC_RAG_OVERFLOW_URL is no longer needed on the game server.

Verification: .venv\Scripts\python -m pytest -q covers capacity routing/recovery, cloud limits,
background-local routing, queue deadlines/cancellation, cooldown, payload/authentication/redaction,
refusals, guarded retries, and retrieval/companion context with mocked providers. Live API timing
still needs a configured key and a controlled conversation test.
API references: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and
[Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

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
| `persona` | Optional, up to 1500 characters. Replaces `data/persona.md` for this call, so a companion answers in its own voice. The leak guard checks this persona instead. |
| `category` | Optional `game`, `general` or `chat` (`chat` means `general`). The caller already classified the message, so the 9091 step is skipped and `method` is `caller`. |

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
- `method` is `social` for an exact social shortcut, `llm` when the model label decided (possibly cached), `wiki` when strong evidence decided early or the model was unavailable, or `caller` when the request's `category` decided.
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

### Companions: `POST /v1/companion/chat` and `/v1/companion/lore` (2026-10-12)

The online companions (Dame Elowen, Mira, Astra, Rosalind, Sable, Vex, Wick) use their own pipeline,
`npc_rag/companion/`. It is built with **LangGraph** (`langgraph` in requirements.txt). LangChain is not used.

```
START ─┬─ classify ── wiki ───────┐
       └─ embed ──┬─ lore ────────┼─ compose ─┬─ generate ─ finish
                  └─ memories ────┘           └─ (game question with nothing to go on: no model call)
```

- **classify, wiki, generate:** reuse Pip's classifier, wiki index and guard. Names from the lore and memories count
  as known, so they pass the guard.
- **lore:** finds this companion's backstory sections. Each section can be locked behind a minimum `bond` and/or a
  story flag. Locked sections are never even scored.
- **memories:** ranks the memories the game server sends: 0.6 × meaning + 0.25 × importance + 0.15 × freshness, above
  `COMPANION_MEMORY_MIN`.
  - While other players can hear (`audience > 0`), embarrassing memories stay out. They only appear as a **slip**,
    which the game server rolls and sends.
  - Anonymous **gossip** is used only when no shared memory fits.
- **finish:** checks the reply really mentions the slip (shared meaningful words) before returning
  `slip: {memory_id}`, which the game server charges Dignity for.

**npc-rag stores nothing about players.** The game server sends everything per call: persona, voice lines, history,
bond, flags, memories, summary, gossip, audience and slip. The only thing kept on a node is the lore the game server
pushes (`POST /v1/companion/lore {version, sections}`).
- A chat whose `lore_version` doesn't match gets **409**, and the game server re-pushes and retries.
- If the main model is down, the endpoint answers **503**, so the game server's pool tries the other node.

Settings (`.env`):

| Setting | Default | Meaning |
|---|---|---|
| `COMPANION_LORE_K` | 2 | Backstory sections per reply. |
| `COMPANION_LORE_MIN` | 0.5 | Relevance a section needs. |
| `COMPANION_MEMORY_K` | 4 | Shared memories per reply. |
| `COMPANION_MEMORY_MIN` | 0.45 | Relevance a memory needs. |
| `COMPANION_GOSSIP_MIN` | 0.45 | Relevance a gossip line needs. |
| `COMPANION_MAX_REPLY_CHARS` | 240 | Reply length cap (area chat). |
| `COMPANION_EMBED_CACHE` | 5000 | Memory texts whose vectors are kept between calls. |

Tests: `tests/test_companion.py`.

**Kinds and speed (2026-10-13, COMPANION_LIFELIKE_PLAN.md phase A).**
- **`kind` on `/v1/companion/chat`:**
  - `chat` (default) and `emote`: the player did something rather than said it.
  - `reaction`: uses `about`, what just happened.
  - `musing`: nobody spoke.
  - `banter`: `partner = {name, persona, line}`. It never carries player memories, gossip or slips.
  - Non-chat kinds skip the classifier.
  - `situation` (up to 10 short facts) adds a "What you notice right now" block.
- **Prompt order is cache-friendly.** The companion's static block (persona, voice, rules) comes first and never
  contains the player or the audience, so llama.cpp reuses its cached prompt between players. Prompt reading is most
  of a reply's time on our cards.
- **Companions get the top `COMPANION_NOTE_K` (2) wiki notes, cut to `COMPANION_NOTE_WORDS` (120),** and generate at
  most `COMPANION_MAX_TOKENS` (100). Pip is unchanged.
- **Copy guard (2026-10-13):** small models ignore "don't repeat". In the continuity test, Qwen3-4B answered a hug with
  a voice line and parroted a banter partner. The graph now checks the reply in code: 7 words in a row from a voice
  example, or 5 from the partner's line, trigger one retry that names the copy. Emote and reaction prompts also
  spell out "respond to that action" and "name what happened".
- **Banter prompt (phase D):** it says not to repeat example lines or the partner's words, and gives topics (their
  past, the place). A live run on 2026-10-13 had Wick reciting a voice line and Sable parroting her before this
  change.
- **`LLM_URL` may list several model copies** (`http://127.0.0.1:9090,http://127.0.0.1:9091`, e.g. one per graphics
  card). Each reply goes to the least busy copy and skips one that fails.

## How each step works

**Classifier** (`npc_rag/classifier.py`). The model on `CLASSIFIER_URL` gets a short instruction and answers GAME or CHAT (4 tokens, temperature 0). From the development PC, Llama-3.2-3B on MONOLITH-III classified 96.2% of the 80 held-out messages in `data/eval_messages.json` correctly, taking about 0.6 s each. Wiki retrieval runs alongside it: a best score of `RAG_GAME_EVIDENCE` (0.66) or more immediately makes the message a game question. Otherwise the model decides; if unavailable or invalid, weak wiki evidence counts as chat. In the service, `CLASSIFIER_TIMEOUT` bounds model queueing plus HTTP time, not retrieval or the whole conversation. Measure accuracy with `python -m npc_rag eval`, or `eval --wiki-only` for the fallback alone.

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

## The in-game NPC

The game side is built (2026-09-25):

- **Lidollquest-server** (`server/tutor.mjs`) places Pip in the five starting lobbies, stores each question from the
  `tutor_ask` command, and forwards it to `POST /v1/npc/chat` on port 9092 in the background with `X-Api-Key`. It uses
  the character id as `player_id` and the character's name as `player_name`, and limits each character to one
  question at a time, a 4 s pause and 100 questions per account per day. Configure it with `NPC_RAG_URL` and
  `NPC_RAG_KEY` in the game server's env; see its README, "Tutor NPC (Pip)".
- **The game client** turns the chat box into a "To Pip" box after **Ask a question.** and shows the answer in the
  dialogue box and the action log.
- **The `/gm` panel** has a **Tutor NPC** tab: on/off, name, greeting, npc-rag health and the latest questions.

The game server must be able to reach this machine on port 9092: run `ps\Register-NpcRagTasks.ps1 -AllowFrom <game server IP>`.

**Two or more AI machines (2026-10-12):** the game server can load-balance several npc-rag nodes (`NPC_RAG_NODES` in its env, `server/ai-pool.mjs`). Install npc-rag on every machine the same way, with the same `NPC_API_KEY`, and run `Register-NpcRagTasks.ps1 -AllowFrom <game server>` on each. The game server sends `history` with every request, so npc-rag's built-in session memory is not used, and any node can answer any player. The current nodes are 47.51.162.106 and 192.168.1.188.

## Separate GM handbook assistant

The editor can open a staff chat window backed by a separate GM wiki index and service on port 9093. All questions are proxied by the game server, preserving the AI server's public-IP allowlist. See [GM_HELP.md](GM_HELP.md) for setup, service keys, firewall rules, screenshots, index refresh and tests. The player service on 9092 is unchanged.
