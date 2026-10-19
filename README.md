# npc-rag: the LiDollQuest tutorial NPC

## Ubuntu 26.04 nodes: port layout, control page and install (2026-10-19)

Linux AI nodes run **Ubuntu 26.04 LTS** (target: 26.04.1). Fedora is no longer supported for AI nodes. The game server's own host is a separate machine and is unaffected.

**Ports.** The control page took 9091, so the layout moved:

| Port | What | Reachable from |
|---|---|---|
| 9090 | kept free (old Sakura copies on Windows; Cockpit if you ever install it) | — |
| 9091 | **npc-rag control page** (`python -m npc_rag admin`) | the LAN |
| 9092 | npc-rag API (the game server calls this) | the LAN |
| 9100+ | the student llama-server copies (`STUDENT_HOST=127.0.0.1`) | this machine only |

ufw opens only 9091 and 9092. Copies never take 9090, 9091 or 9092, even with an old `STUDENT_BASE_PORT=9090` (`STUDENT_RESERVED_PORTS` + `NPC_PORT` + `ADMIN_PORT` are skipped). The game server no longer has to reach a model port: `/health` reports `capacity.slots` (every local copy's slots added up) and the pool sizes itself from that.

**What changes on Windows nodes after a pull:** the copies move to `127.0.0.1:9100+` too. Sweeps from another machine (`llama_concurrency_sweep.py http://<node>:9090`) stop working; run the sweep on the node itself against `http://127.0.0.1:9100`, or use the control page's **Run sweep** button. Set `STUDENT_HOST=0.0.0.0` and `STUDENT_BASE_PORT=9090` in `.env` to keep the old behaviour.

**The control page (9091).** It runs as its own service (`npc-rag-admin.service`), so it stays up while npc-rag restarts. It shows:
- tiles for npc-rag, the main model, reply slots, the student model, copies alive, the Decider, the systemd state and the restart count;
- one row per student copy: card, port, alive, restarts, last exit code, requests in flight, recent seconds per reply;
- the machine: load, free memory and disk, GPUs from `rocm-smi` (or `nvidia-smi`);
- the npc-rag journal tail and the raw `/health`.

It can:
- **Edit settings.** Only a whitelisted set of keys (student fleet, model limits, Decider, overflow, secrets). Secrets are write-only: the page only says whether they are set. Every save runs npc-rag's own settings check (`load_settings`) on the new file in a clean subprocess first; a bad value is refused and the file is left alone. The write is atomic, with a timestamped `.env.bak-*` beside it (newest 5 kept). Host, port and the admin password can't be changed from the page, so you can't lock yourself out.
- **Restart npc-rag** with `systemctl --no-block restart npc-rag.service`. The polkit rule in `linux/` allows exactly that unit (start/stop/restart) for the `npcrag` user. On Windows, set `ADMIN_RESTART_CMD`.
- **Run a speed sweep:** 1, half and all slots on every live local copy; tokens/s and seconds per reply. Players share those slots, so run it when the node is quiet.
- **Check & install updates:** fetch the checkout's tracked branch, install newer commits, then restart RAG and the control page. Progress and the previous/target revisions stay visible; sign in again after the control page reloads.

For an existing Ubuntu installation, first bring this code onto the node, then run
`sudo bash /opt/npc-rag/linux/install-updater.sh` once. It installs the dedicated
`npc-rag-update.service` and the narrow polkit permissions, and restarts only the
control page. It does not rebuild Python, ROCm or the student models. New installs
through `linux/install-ubuntu.sh` include the updater automatically.

The updater is `python/update_repo.py`. Manual commands, run as the checkout owner:

```bash
cd /opt/npc-rag
sudo -u npcrag .venv/bin/python python/update_repo.py --check
sudo -u npcrag .venv/bin/python python/update_repo.py --install --restart-services
```

The button uses the existing admin login and CSRF header; it cannot select arbitrary
commands, remotes or branches. A separate oneshot service survives both web-service
restarts. Settings writes, restarts and speed sweeps are blocked while that update job
runs. `data/repo-update.json` stores its progress/result, and the admin audit records
who requested it. Inspect `journalctl -u npc-rag-update.service` for job status.

Updates are fast-forward only. Tracked local code edits (including a manually applied
Decider patch), local commits ahead of upstream, divergent history, or untracked files
that would be overwritten stop installation; there is no reset, stash, clean or forced
checkout. Resolve those edits deliberately before updating. Legacy tracked
`data/admin-audit.jsonl` additions are preserved, and upstream replacements of that file,
the node's `.env`, models, indexes or venv are refused. The `npcrag` user needs a writable
checkout and Git access to its configured upstream; no credentials are accepted through
the page. Commit/publish changes to the upstream branch before expecting nodes to find them.

Ordinary code updates leave installed packages alone. Changed `requirements.txt` or
`requirements-decider.txt` files trigger dependency resolution and installation with the
existing `llama-cpp-python` version constrained and without upgrade/force-reinstall flags.
This preserves the custom ROCm/CUDA/Vulkan build; a conflicting native-version change
requires manual backend installation. Changes to systemd units or polkit rules likewise
require a manual update and `linux/install-ubuntu.sh`, because the updater runs unprivileged.
Dependency or readiness failures are reported with the old/new revisions and do not
silently reset code or roll back packages. After restart, readiness must report a healthy
RAG index, reply model, classifier and every managed student copy. A failed update may
have installed code already; inspect its result before retrying.

Validation: `.venv/bin/python -m pytest tests/test_repo_update.py tests/test_admin.py -q`
uses disposable Git remotes and mocked service actions. It covers checks, no-op updates,
fast-forwards, conflicts, protected data, locking, native dependency preservation,
authentication/CSRF, duplicate jobs and interrupted status. Real Linux systemd and ROCm
readiness still require acceptance on the deployed node.

**Security.** Sign-in is `ADMIN_PASSWORD` only, never `NPC_API_KEY`. The page refuses to start without 8+ characters. The session cookie is HMAC-signed, HttpOnly, SameSite=Strict, and lasts 12 h (restarting the page signs everyone out). Five wrong passwords per address per 5 minutes means a 429. Every POST also needs the page's own `X-NPC-Admin` header, which a cross-site form can't send. Each action is logged to `data/admin-audit.jsonl` with key names only, never values. The page is plain HTTP: keep 9091 on the LAN (or put it behind TLS).

**Install on Ubuntu 26.04** (as root):
```bash
git clone https://github.com/AELovelace/npc-rag.git /opt/npc-rag
bash /opt/npc-rag/linux/build-llama.sh     # llama-server: ROCm HIP (gfx1030) + RADV Vulkan into /opt/llama.cpp
bash /opt/npc-rag/linux/install-ubuntu.sh    # user, Python 3.12 venv (+ Decider), .env, systemd units, polkit rule, ufw
```
`build-llama.sh`:
- gets ROCm from Ubuntu's own archive (`apt install rocm`). That was ROCm 7.1 at 26.04's launch, a few releases behind AMD.
  - For newer ROCm, run AMD's installer first, then `sudo ROCM_SOURCE=amd bash build-llama.sh`.
  - If HIP can't build, it says so and still builds Vulkan. npc-rag runs fine on Vulkan alone.
- `BUILDS=vulkan` skips HIP. `LLAMA_TAG` (default `b9481`) and `GPU_TARGETS` (default `gfx1030`, the 6800 XT) pick the build.

`install-ubuntu.sh`:
- creates the `npcrag` user in the render, video and systemd-journal groups;
- builds `.venv` with **Python 3.12**. 26.04 ships 3.14, and the Decider stack (torch, llama-cpp-python) is tested on 3.12. So the script installs `uv` from PyPI into `/opt/uv/bootstrap`, and `uv` fetches 3.12 into `/opt/uv/python`. The system Python is left alone. Set `PYTHON_VERSION` to try another;
- writes the node keys into `.env` (`STUDENT_FLEET=on`, `STUDENT_HOST=127.0.0.1`, `STUDENT_BASE_PORT=9100`, `CLASSIFIER_BACKEND=decider`, and a generated `ADMIN_PASSWORD`, printed once);
- installs `linux/npc-rag.service` and `linux/npc-rag-admin.service`, plus the polkit rule;
- allows 9091 and 9092 in ufw. If ufw is off, it says how to enable it without locking out SSH (`ufw allow OpenSSH && ufw enable`).

It is safe to run again: an existing `.env` keeps its own choices. `KillMode=control-group` means stopping npc-rag stops every copy. Outside systemd, each copy also gets `PR_SET_PDEATHSIG`, so it dies with npc-rag. On Linux the auto-find prefers a HIP build over Vulkan (Windows prefers Vulkan); set `LLAMA_SERVER` to pin one. Then set `NPC_API_KEY` on the control page to the game server's `NPC_RAG_KEY`, and restart.

**Don't forget:**
- `CLASSIFIER_BACKEND=llamacpp` would now point at the control page (its old default URL is `:9091`). Student-fleet nodes use `decider`.
- The GM handbook's `GM_LLM_URL` default is `:9090`, which nothing serves on a student-fleet node. Point it at `http://127.0.0.1:9100`.
- The scripts in `linux/` must keep LF line endings (`.gitattributes` pins them). A CRLF breaks bash and systemd.

## Student fleet: npc-rag runs its own lidoll-fm copies (2026-10-18)

`STUDENT_FLEET=on` makes npc-rag start its own llama-server copies of the companion model, instead of you starting them by hand in Sakura.

**What it does when npc-rag starts:**
1. **Gets the model.** It downloads the release named in `npc_rag/student_release.json` from https://lidoll.dev/gguf/. The default is `lidoll-fm-0.4-Q8_0.gguf`. It checks the size and sha256, then keeps the file in `data/models/<release>/` and reuses it offline from then on.
2. **Finds the GPUs** with `llama-server --list-devices` (or uses `STUDENT_DEVICES`).
3. **Starts `STUDENT_COPIES_PER_DEVICE` copies on every card,** on free ports from `STUDENT_BASE_PORT` (default 9100, listening on `STUDENT_HOST`, default 127.0.0.1). Each copy runs with:
   - `-np STUDENT_SLOTS`
   - `-c` = slots × `STUDENT_CTX_PER_SLOT`
   - `--n-gpu-layers 999 --device <card>`
   - plus `STUDENT_ARGS` (default `--flash-attn on --cache-ram 4096`)
4. **Waits until every copy's `/health` answers,** then points itself at them: `LLM_URL` becomes all the copies, and `LLM_CONCURRENCY` becomes `STUDENT_SLOTS`.
5. **Keeps them running.** A watchdog restarts any copy that exits, waiting longer after each crash, up to 60 s. On Windows the copies sit in a job object, so they die with npc-rag, even after `Stop-Process -Force`. `data/student-fleet.json` cleans up any leftovers on the next start.
6. **Reports on `/health`,** under `student_fleet`: the model, ports, cards, whether each copy is alive, and its restart count. `capacity.slots` is the total reply slots; the game server's pool reads it instead of llama.cpp's `/props`.

**How npc-rag shares replies across copies:** each reply goes to the copy with the lowest (requests in flight + 1) × recent seconds per reply. This is the same idea as the game server's pool between nodes. A faster card therefore takes more of the work, and a slower one still helps. Plain "least busy" gave node A's ~4× slower card 0 the same share as card 1, which created the slow tail (`llm.py` `LlamaPool`, 2026-10-18).

**Why several copies per card:** llama-server feeds its GPU from **one CPU thread**. On node A's RX 6800 XTs (Vulkan), that thread was pinned at 100% while the GPU waited, which capped one copy at about 90 generated tokens/s however many slots it had. Each extra copy brings its own thread.

**What to avoid:**
- **Keep the RAM prompt cache on** (`--cache-ram 4096`, the default). Every companion and resident has a long, fixed prompt start, and the cache restores it into any slot. With the speed-aware pool, it took the 100-player test from 4.7–5.7 s to 3.7–4.4 s typical (2026-10-18).
- **Don't compress the KV cache** (`--cache-type-k/v`). It didn't help speed, and a 1.2B model's cache is tiny anyway.
- **Measure before and after a change** with lidollquest `python/llama_concurrency_sweep.py http://127.0.0.1:9100` on the node itself (about 20 s), or the control page's **Run sweep**.

**Switching a node over:**
1. Stop Sakura's 9090/9091 copies.
2. In `.env`, set `STUDENT_FLEET=on`. `LLAMA_SERVER` is optional: if it's empty, npc-rag looks for llama-server on PATH, then in `C:/Scripts/llama/*/`, the Sakura folder and `/opt/llama.cpp/*/bin` (Vulkan preferred on Windows, HIP on Linux). For Titan X nodes, set it to the CUDA build.
3. Run `ps\Start-NpcRag.ps1`. The first start downloads the model (1.25 GB).

**Updating or rolling back:** add a new release to `student_release.json` (file, bytes, sha256, download_url) and make it the `default`. To roll back, set `STUDENT_MODEL=lidoll-fm-0.3`.

## Authorized v3 rollout (2026-10-15)

The user accepted the known opening miss and selected v3 for rollout with v2 rollback.
The exported Q4_K_M comparison is routing 97/100 (unchanged), retained participation
109/112 versus 106/112, new cases 63/64 versus 57/64, and reconstructed hired social
openings 20/20 versus 3/20. One new keepsake-box opening still gets QUIET (7/8 pass).
The failed behavior gate remains recorded; live shared-pool load testing is not claimed.

Upload `decider-multitask-v3.gguf` and `decider-multitask-v3-support.zip` from
`runs/releases/decider-multitask-v3-q4km/` to `https://lidoll.dev/gguf/`.
Install the updated game-server participation validator first (it accepts v2 and v3),
then the updated npc-rag code on every RAG host. Set:

```dotenv
CLASSIFIER_BACKEND=decider
DECIDER_MODEL=lidoll/decider-multitask-v3
DECIDER_GGUF_FILE=decider-multitask-v3.gguf
PARTICIPATION_ENABLED=true
PARTICIPATION_MODEL=shared
```

Keep existing GPU, worker, timeout and ambient-reserve settings. Restart RAG and check
both tasks are healthy, report the v3 release and share worker PIDs. The managed loader
pins v3 assets in its own cache; v2 keeps its original pins/configs and cache. Roll back
with `DECIDER_MODEL=lidoll/decider-multitask-v2` and
`DECIDER_GGUF_FILE=decider-multitask-v2.gguf`, retain `PARTICIPATION_MODEL=shared`, restart.
Packaging authorization does not mean the assets have been uploaded or hosts restarted.

Repackage with `python python/package_multitask_release.py --accept-known-opening-miss`.
The historical v2 section below describes the previous release; use this section for v3.


## Targeted hired-opening v3 experiment (2026-10-15)

This candidate addresses the live v2 failure where a hired companion stays quiet on an
opening social question with no conversation history. It preserves every v2 row/split
and adds 40 authored training families (320 rows), plus 8 families each for validation,
development and test (64 rows per split). Each family has four JOIN and four QUIET
contexts: hired opening, hired/nonhired ongoing exchange and explicit address versus
an unaddressed bystander, another addressee, quoted speech and requested silence.
All new rows are synthetic drafts, not reviewed real-player exports. Family variants
are correlated; fresh held-out wording still shares context templates with training.

Participation has 1,120 train / 136 validation / 136 development / 176 test rows.
Routing retains 800 training rows. The same original pinned base, rank-8 QLoRA,
two epochs and equal aggregate task loss are retained (240 optimizer steps).
`python/build_participation_v3.py` regenerates the dataset and
`python/check_participation_v3.py` checks both task prompts and replay exclusion.
The replay reconstructs 50 gate inputs from 25 provably empty-history events in the
saved 100-player ramp; 20 hired/social inputs have the explicit JOIN expectation.
The missing original quiet-analysis script and unsaved generated history are not
replaced with invented dialogue. Catalog provenance limits are recorded in the
replay manifest. Unlabelled bystander/game inputs are diagnostics, not accuracy golds.

`ps/Train-MultitaskV3.ps1` runs training, specialist comparison, merge, Q4_K_M export,
and v2/v3 evaluation on exactly the same expanded test cases. Then
`python/report_multitask_v3.py` scores both exports on the frozen opening regression
and writes `targeted-comparison.json` with retained v2, new v3, scenario, routing and
boundary results. Behavior gates require successful hired openings, preserved ongoing
conversations, no new boundary/bystander interruptions, and no aggregate regression
on routing or retained participation tests. A failed gate stays visible; no automatic
deployment or test-driven retraining occurs.

Candidate paths: `runs/multitask-v3-hired-opening-20261015` and
`runs/releases/decider-multitask-v3-q4km`. The separate pipeline status is
`runs/multitask-v3-hired-opening-20261015-pipeline.json`; stdout/stderr logs share that
run prefix. Existing live v2 settings, release pins and assets are not changed.



## Combined Decider v2 upload and shared workers

The prepared runtime uses `lidoll/decider-multitask-v2` for GAME/CHAT and JOIN/QUIET
through **one model per selected GPU**, with a shared admission queue. CPU mode uses
one model and one native worker. Routing keeps its state-first 384-token prompt;
participation keeps its schema-first 768-token prompt. Over-budget inputs fail rather
than truncate. Confidence remains uncalibrated. Existing specialist aliases still work
for rollback; HTTP classifier mode remains available.

Upload these two files from `runs/releases/decider-multitask-v2-q4km/` to
`https://lidoll.dev/gguf/`:

- `decider-multitask-v2.gguf` (1,274,396,192 bytes, Q4_K_M)
- `decider-multitask-v2-support.zip` (matching tokenizer, configs, license, provenance
  and evaluation summary; approximately 3.9 MB)

`SHA256SUMS.txt` contains both hashes. The BF16 intermediate is not an upload asset.
The ZIP contains model support files; install the updated **npc-rag code** on both
RAG servers as well. Deploy the prepared **Lidollquest-server participation validator**
before switching RAG: the old validator only accepts the specialist's model identity.
After uploading, set these values in each server's active `.env`
(explicit old environment values override the new defaults):

```dotenv
CLASSIFIER_BACKEND=decider
DECIDER_MODEL=lidoll/decider-multitask-v2
DECIDER_GGUF_FILE=decider-multitask-v2.gguf
PARTICIPATION_ENABLED=true
PARTICIPATION_MODEL=shared
PARTICIPATION_TIMEOUT=10
```

Keep each host's existing `DECIDER_GPU_LAYERS`, `DECIDER_GPU_DEVICES`, and
`DECIDER_THREADS`, then restart RAG. Startup verifies/downloads the pinned model and
support files into `data/models/decider-multitask-v2`; configs come from the checked-in
`npc_rag/multitask_config`. For an offline install, extract the ZIP and place the GGUF
in that same directory. Startup decodes both tasks before becoming ready. No new port
or game-server endpoint is required; keep `QUEST_PARTICIPATION_BACKEND=decider`.

On `/health`, `classification.shared_pool` and `participation.shared_pool` must be
true. In GPU mode both sections list **the same worker PIDs**, with one per selected
GPU; their capacity/busy/queued counts describe the same pool and should not be added
together. An abandoned request retains its native slot until inference finishes, and
a timed-out queued request never gets submitted. Participation shutdown does not free
the routing model; the routing owner drains and closes the shared pool once.

To roll back, set `DECIDER_MODEL=lidoll/decider0-1`,
`DECIDER_GGUF_FILE=decider0-1.gguf`, and
`PARTICIPATION_MODEL=lidoll/decider-chat-0-1`, then restart. That loads the previous
two specialist pools again. Preparation does not upload assets or restart live hosts.

Rebuild the support ZIP/pins with `python/package_multitask_release.py`. Run
`tests/test_multitask_runtime.py` with the existing Decider/pool/participation/service
tests for admission, cancellation, task identity, archive checks, and lifecycle checks.
`python/check_multitask_serving.py` replays all 212 held-out decisions through the real
shared CPU serving interfaces and writes `serving-check.json` beside the release.
Verification on 2026-10-15: all 212 labels matched the offline evaluation, with one
loaded model (routing 97/100, participation 106/112). The npc-rag suite passed 265 tests
with 3 optional tests skipped; the server's 28 participation/companion/ambient checks
passed. ZIP extraction/hash checks and cached offline installation passed. These local
checks do not measure live GPU shared-pool latency under load.
The earlier experiment notes below record the training/evaluation history; the shared
runtime described here supersedes their statements that serving integration was pending.

## Combined Decider experiment (2026-10-15)

Expanded v2 run: `runs/multitask-v2-first-pass-20261015` starts from the same original
pinned base with **800 routing + 800 participation training rows**, equal task loss weights,
two epochs and 200 optimizer steps. `python/build_participation_v2.py` adds 200 authored
scenario families across expertise, follow-ups, conversation handoffs, farewells, direct
address, invitations, OOC, quotations, silence/privacy boundaries and emotes. Each new
family has paired JOIN/QUIET contexts in NPC and companion roles; the 800 new rows are
related variants, not 800 independent conversations. There are no reviewed live exports
in this dataset. All original v1 rows and splits remain unchanged.

Participation splits are 800 train / 72 validation / 72 development / 112 test, with
80 newly held-out test rows from 20 families. V1 tests are retained regression checks;
the new synthetic cases still share broad categories with training. The manifest is
`data/participation/manifest-v2.json`. Current data checks reject cross-task utterance
overlap and conversation-family leakage. The full tokenization check found no truncation
(largest training prompt: 215 tokens).

V2 completed successfully: 200 steps in 2,666.24 seconds of training/final scoring,
2.85 GiB peak PyTorch allocation. Development-based selection retained epoch 1 (both
epochs tied on recall/accuracy; epoch 1 had lower development loss). Its adapter scored
98/100 routing and 110/112 participation; the exported Q4_K_M scored 97/100 and 106/112.
The completed same-data GGUF comparison improved participation from v1's 98/112 (87.5%)
to 106/112 (94.6%), with unwanted interruptions falling from 10 to 2 and missed joins
remaining at 4. On the 80 new participation cases alone, accuracy rose from 66/80 to
74/80; both models retained 32/32 on the original cases. Synthetic routing improved
96/100 to 97/100, and fixed raw routing improved 91/96 to 92/96. All v2 evaluated
per-class gates passed. These are synthetic/offline results with uncalibrated confidence,
not live shared-pool qualification. The report paths below contain the full decisions;
production models remain unchanged.

```powershell
.venv/Scripts/python.exe python/build_participation_v2.py
powershell -ExecutionPolicy Bypass -File ps/Train-MultitaskDecider.ps1 -ParticipationDataset data/participation/synthetic-expanded-v2.jsonl -ParticipationDevDataset data/participation/development-v2.jsonl -Output runs/multitask-v2-first-pass-20261015
powershell -ExecutionPolicy Bypass -File ps/Finish-MultitaskDecider.ps1 -Run runs/multitask-v2-first-pass-20261015 -ExpandedParticipation -Name decider-multitask-v2 -BaselineRelease runs/releases/decider-multitask-v1-q4km
```

The finish workflow compares the routing specialist on its unchanged test set and the
participation specialist only on the preserved v1 subset, after auditing original source
hashes and unchanged rows/splits. It also exports/evaluates v2 and evaluates the **previous
combined GGUF on the same expanded datasets**, writing `previous-model-on-current-data.json`
inside the v2 run. Each GGUF report separates original and new participation test cases.
Use fresh output paths for additional runs; nothing is deployed by this workflow.

First run completed: `runs/multitask-v1-first-pass-20261015` selected epoch 2 after
120 optimizer steps (1,530.72 seconds of training/final scoring; 2.82 GiB peak PyTorch
allocation). On identical synthetic tests, the combined adapter scored routing **96/100**
versus the specialist's **99/100**, and participation **32/32**, matching its specialist.
Participation validation was **28/32** and development **30/32**, so the perfect small
test set does not establish broad reliability. The local Q4_K_M export reproduced both
synthetic test scores through one loaded model instance. Raw fixed routing scored
**91/96** with five missed GAME questions (GAME recall 43/48, below the 90% gate).
That raw score is not directly comparable to previous wiki-assisted routing reports.
See `specialist-comparison.json` in the run and
`runs/releases/decider-multitask-v1-q4km/quantized-multitask.json`. The experiment is
complete but remains uncalibrated and not deployment-qualified; live models were not changed.
The evaluator's initial metadata-as-example error was fixed before the successful rerun;
the fixed routing loader now ignores `_about` and tests enforce exactly 96 labeled examples.

`ps/Train-MultitaskDecider.ps1` trains one fresh QLoRA adapter on both GAME/CHAT and
JOIN/QUIET, starting from the same pinned original Decider-2B base. It uses cached
weights only. Routing keeps its state-first prompt and 384-token budget; participation
keeps its schema-first prompt and 768-token budget. The 800 routing and 160 participation
training rows are shuffled together, with loss weights 0.6 and 3.0 respectively so each
task has equal total influence. No evaluation rows are oversampled or used for training.
Existing task-family checks also run, plus a cross-task check against held-out utterances.

```powershell
powershell -ExecutionPolicy Bypass -File ps/Train-MultitaskDecider.ps1 -CheckData
powershell -ExecutionPolicy Bypass -File ps/Train-MultitaskDecider.ps1 -SmokeTest
powershell -ExecutionPolicy Bypass -File ps/Train-MultitaskDecider.ps1 -Output runs/multitask-v1-first-pass-20261015
.venv-train/Scripts/python.exe python/compare_multitask_runs.py --run runs/multitask-v1-first-pass-20261015
```

The smoke test covers eight rows **per task**, four optimizer steps, and matching
saved/reloaded probabilities. The full run uses two epochs and 120 optimizer steps.
Checkpoint selection considers the worst of all four development class recalls, then
equal-task mean accuracy and loss. Reports contain `by_task` metrics, including unwanted
interruptions and missed joins. The held-out synthetic test runs only after selection.
`compare_multitask_runs.py` checks source hashes, policies, base revision and test IDs
before comparing with the previous specialist runs. It reports regressions for each task.

This is an experiment using existing synthetic data, with uncalibrated confidence.
Training produces `best-adapter/`; it does not change the live model configuration.
The distinct multi-task policy is deliberately rejected by existing single-task loaders.
To produce and test a local experimental GGUF after a completed full run:

```powershell
.venv-train/Scripts/python.exe python/merge_classifier.py --task multitask --run runs/multitask-v1-first-pass-20261015 --local-files-only
.venv-train/Scripts/python.exe python/export_classifier_gguf.py --task multitask --run runs/multitask-v1-first-pass-20261015 --name decider-multitask-v1 --output runs/releases/decider-multitask-v1-q4km
.venv-train/Scripts/python.exe python/evaluate_multitask.py --release runs/releases/decider-multitask-v1-q4km
```

`ps/Finish-MultitaskDecider.ps1 -Run runs/multitask-v1-first-pass-20261015` performs that
comparison/merge/export/evaluation sequence and records its stage in `experiment-status.json`.
When training is already running in a background process, `-WaitForProcessId <pid>` waits
for that process first (at most four hours). Each stage stops on errors, and existing
merge/export directories are preserved. An accuracy regression is recorded as an experimental
result; it does not prevent local export for inspection and never triggers deployment.

The offline GGUF evaluator alternates both tasks through **one loaded model instance**,
then runs the fixed routing messages without wiki evidence. This does not establish
live shared-worker capacity or calibrated probabilities. Production still needs shared
pool integration and real exported/reviewed participation decisions before qualification.
Relevant checks: `.venv/Scripts/python.exe -m pytest tests/test_multitask_training.py
tests/test_training_data.py tests/test_participation_training.py tests/test_training_report.py -q`.

Overflow-enabled comparison (2026-10-14): repeating the same scheduled-player workload
with `--overflow enabled` passed at 25 and 50 active users. Generated median/p95 seconds:
25 users 2.355/4.057; 50 users 4.493/9.055; 75 users 5.409/13.551. The 75-user stage
failed only latency limits; 100 was skipped. All three ten-minute stages completed:
3,000 messages, 2,936 generated replies, 60 source/content-guard replies, four quiet
decisions, zero availability failures. Highest passing tested population was 50 with
overflow versus 25 local-only, not an exact maximum or total gameplay capacity. Cloud
requests were observed in health snapshots with no sampled cooldown; exact provider
usage was not recorded. Both hosts finished idle/healthy, all eight classifiers had
zero native failures, and all four generation endpoints answered health. Overflow remains
enabled. See server `artifacts/overflow-player-capacity-20261014.json` and
`artifacts/player-capacity-local-vs-overflow-20261014.json` for evidence and comparison limits.

Local-only player test (2026-10-14): both hosts kept cloud overflow disabled throughout
37 recorded health snapshots. At one message per player every 30s, 25 active users passed
ten minutes: 500 messages, 483 generated replies, one availability fallback (0.2%),
generated median/p95 2.441/4.340s and stable backlog. Fifty active users overloaded the
pool: arrivals stopped around 362s, with 149 failures among 610 offered messages after
draining; generated median/p95 was 6.030/19.289s. Stages 75/100 were skipped. All eight
classifier workers finished healthy/idle with zero native failures, and all four
generation endpoints answered health afterward. This establishes 25 as the highest
passing tested active population for this workload, not an exact maximum or online-player
limit. See server `artifacts/local-player-capacity-20261014.json` and its README for
the scheduled arrivals, histories, nearby gate fanout, burst behavior and caveats.

For a local-only capacity test, temporarily set `OPENAI_API_KEY=` in each RAG host's
active `.env` and restart RAG; preserve the key privately for restoration afterward.
Clear any process/service environment override as well, since it takes precedence over
the file. `/health` must report `overflow.enabled: false` on **both** hosts. An empty key
constructs only the local model client; no OpenAI client is created. The server checkout's
`deploy/ai-player-test.mjs` verifies this before and throughout its scheduled-player test.
See the server README for workload, pass criteria and command. Removing the key from the
test runner's environment alone does not disable overflow on remote RAG workers.

## Decider Chat 0.1 upload and B-test deployment

Live B completed (2026-10-14): both RAG hosts ran two routing and two participation
workers each, all healthy and idle afterward with zero native failures. The standalone
60% companion / 25% Pip / 15% gate simulation completed 719 requests, zero HTTP errors
and zero availability fallbacks. At concurrency 1/2/4/8/16/24, generated-reply medians
were 1.447/1.827/2.409/3.324/3.821/1.729 s. Participation gate medians were
0.312/0.311/0.381/0.444/0.471/0.636 s; historical Qwen A was
0.175/0.146/0.210/0.552/1.710/2.186 s. The gate improved under load; dialogue did not
uniformly improve, and its 24-concurrency p95 was still 15.825 s. Random workload,
cache state and enabled cloud overflow prevent a controlled model comparison. This
checks load/stability, not participation accuracy or the updating live game server.
See server `artifacts/decider-chat-0-1-simulated-b-20261014.json`, adjacent
`-health-after.json`, and `decider-chat-0-1-simulated-ab-20261014.json`.

Prepared Q4_K_M files are in `runs/releases/decider-chat-0-1-q4km/`:
`decider-chat-0-1.gguf` (1,274,396,192 bytes) and `decider-chat-0-1-support.zip`
(tokenizer, matching sidecars, license, checksums and qualification). Upload both beside
the existing routing release at `https://lidoll.dev/gguf/`. SHA256SUMS.txt records their
hashes. Native CPU evaluation scored 31/32 on each of development and test, with one
missed join and zero unwanted joins per split. Each split has only eight synthetic
families; this is an experimental B candidate, not a production behavior guarantee.

After the prepared code is committed and pulled to both RAG hosts, add to their `.env`:

```dotenv
PARTICIPATION_ENABLED=true
PARTICIPATION_MODEL=lidoll/decider-chat-0-1
PARTICIPATION_TIMEOUT=10
```

Restart RAG. The startup check verifies/downloads the separate release and uses configs
from `npc_rag/participation_config` in Git. Model/tokenizer assets are pinned in
`npc_rag/participation_release.json`. No additional port is needed. Existing
`DECIDER_GPU_LAYERS`, `DECIDER_GPU_DEVICES` and `DECIDER_THREADS` apply to this model too:
GPU mode adds one participation worker/model per selected GPU while retaining routing.
Allow memory for both models alongside generation. Every new worker must load and decode
a synthetic input before startup succeeds. Offline restarts reuse verified cached files.

`GET /health` exposes `participation.enabled`, `healthy`, policy and per-GPU workers.
`POST /v1/participation/decide` accepts the bounded context described below, requires the
existing NPC API key, and returns JOIN/QUIET, its uncalibrated score and timing. Disabled,
unavailable or timed-out models return 503; invalid context returns 422. It has no decision
cache, so a repeated message with different conversation context cannot reuse an old label.
Timeouts never free a still-running native worker for overlapping inference.

The updated game server can select this gate with `QUEST_PARTICIPATION_BACKEND=decider`
and a restart; default/rollback is `qwen`. Candidate selection, direct-name bypasses,
cooldowns and existing follow-up availability behavior stay the same for B. Its context
includes only the active speaker/candidate exchange (up to four turns), nearby companion
names and optional `online.participation.role`/`interests` from the catalog. It does not
yet collect other nearby players' conversation history or broaden unsolicited candidates.
No live game-server switch is needed to run the standalone B harness after RAG deployment.
See the server README for its command. Set `PARTICIPATION_ENABLED=false` and restart to
release the extra RAG workers; roll back the game gate first.

## Experimental sociable participation classifier

`python/train_classifier.py --task participation` trains a separate Decider adapter for
JOIN/QUIET: should a nearby companion or NPC speak next? Doll selected a sociable policy:
join direct invitations, ongoing exchanges and open conversation where the character has
relevant knowledge or interests; respect other addressees, requests for quiet and OOC text.
It does not replace the released GAME/CHAT checkpoint or the live Qwen participation gate.

The input contract in `npc_rag/participation_policy.py` contains the candidate's name, kind,
role and up to four interests; player name/message/kind; nearby names; up to four recent
turns; and conversation/hiring flags. Only context enters the prompt. Labels, family IDs
and scenario explanations stay outside it. JSON bounds reject oversized or unexpected
fields. The default layout puts the fixed question/options first (`schema_first`); this
alone does not enable native prefix caching or establish a latency improvement.

The reproducible draft dataset has **256 rows from 64 scenario families**: 160 training,
32 validation, 32 development for checkpoint selection and 32 final test. Each family has
JOIN/QUIET contexts and companion/NPC variants, all kept in one split. Identical messages
can require different decisions when their context changes. Candidate names in evaluation
are separate from training. These are synthetic hypothetical roles, not the production NPC
catalog, and the 32-row test represents only eight scenario families. Review and expand
with recorded, context-rich game conversations before claiming player-facing quality.

```powershell
# Uses the existing separate NVIDIA training environment; the RTX 3080 smoke check works.
powershell -ExecutionPolicy Bypass -File ps/Train-Participation.ps1 -CheckData
powershell -ExecutionPolicy Bypass -File ps/Train-Participation.ps1 -SmokeTest
powershell -ExecutionPolicy Bypass -File ps/Train-Participation.ps1
# Add -LocalFilesOnly when the pinned training checkpoint/tokenizer are already cached.
```

Regenerate the pilot with `.venv/Scripts/python.exe python/build_participation_dataset.py`.
Training uses the pinned original trainable Decider base, not `decider0-1.gguf`. Reports
include unwanted interruptions (QUIET predicted JOIN), missed joins, JOIN precision and
both class recalls. The existing 90% per-class training gate is only a pilot diagnostic,
not release approval. The result is an uncalibrated LoRA adapter under `runs/participation-*`;
the merge/export tools require explicit `--task participation` and a separate release name.
The first GGUF/runtime preparation is described above; live activation remains opt-in.

First completed pilot (2026-10-14): `runs/participation-v1-first-pass-r2-20261014` selected
epoch 2 after 20 optimizer steps. Training/evaluation took 299.86 seconds on the RTX 3080,
with 2.82 GiB peak allocated GPU memory. Development improved from 23/32 to 30/32; both
remaining misses are variants of a dancer failing to join an open dance question. The
untouched synthetic test scored 32/32 (eight families), with zero false joins/missed joins.
Development JOIN recall remains 87.5%, so the pilot does not meet the 90% per-class gate
on that split. The saved artifact is `best-adapter`, with `final-test.json` and
`completion.json` beside it. It is not a GGUF release or a live-service benchmark. Broader
reviewed conversations, context-matched Qwen comparison and native quantized evaluation
are still needed. Current pilot prompts measured 155-176 tokens with the pinned tokenizer.

Broader sociable deployment: the new decision now uses existing RAG HTTP port 9092 with
a separate model identity and matching prompt encoder. To expand unsolicited replies,
the game server must also allow relevant nearby
candidates to reach the model: today's `addressees()` filters discard many unnamed lines
before the Qwen gate. Keep earshot, consent/access settings, cooldown and stale-job checks
in server code. Limit unsolicited replies to one chosen character per message; scoring
every NPC independently without arbitration risks a chorus. Compare against Qwen with
the same context, first in shadow mode, before switching decisions. Measure false joins,
missed invitations, candidate count, gate latency and full replies under concurrent load.

## Live classifier validation

Vulkan recovery verified (2026-10-14): `192.168.1.188:9092` now completes native decisions
on both GTX TITAN X workers (`Vulkan0`/`Vulkan1`). Live routing scored 76/80 held-out (95%)
and 16/16 mixed (100%). The subsequent pooled 100-chat burst generated 100 replies without
HTTP errors or availability fallbacks: median 8.669 s, p95 31.101 s. The Titan node's
reply median was 19.212 s, with classification/generation phase medians 1.358/17.691 s.
Generation is the dominant delay in that burst. The pooled sustained simulation completed
678 requests without HTTP errors or availability fallbacks; generated-reply medians at
1/2/4/8/16/24 concurrency were 1.380/1.391/1.990/3.606/4.948/1.702 s. All four workers
remained healthy with zero failures (AMD completions 59/62; Titan 51/52). Repeated prompts,
warm caches and enabled cloud overflow make this a capacity diagnostic, not a matched
cold-cache A/B or production latency guarantee. The burst still misses the five-second median.
Artifacts: `runs/live-decider01-titan-vulkan-accuracy-20261014.json` and server artifacts
`decider01-all-vulkan-pooled-chat-20261014.json`, `decider01-all-vulkan-simulated-players-20261014.json`
and its adjacent `-health-after.json`. Live health does not yet expose `startup_inference_ms`;
this verifies actual inference, not deployment of the newer diagnostics patch below.

CUDA failure follow-up (2026-10-14): the supplied log identifies both GTX TITAN X cards as
compute capability 5.2 and shows fatal native CUDA errors on their first decode. The exact
[0.3.36-cu124 wheel workflow](https://github.com/abetlen/llama-cpp-python/blob/v0.3.36-cu124/.github/workflows/build-wheels-cuda.yaml#L212)
targets 6.0 and newer architectures and omits 5.2, a concrete compatibility mismatch and
likely cause of the failure. The generic abort alone does not expose the CUDA error code.
Stop RAG on that host, run `ps/Install-Decider.ps1 -Backend Vulkan`, and restart with
`DECIDER_GPU_LAYERS=-1` and default `DECIDER_GPU_DEVICES=auto`. The replacement still needs
successful inference and live validation; those checks subsequently passed as recorded above.

Native warnings/errors now remain visible in stderr instead of being swallowed by Decider's
quiet callback. Each GPU worker also performs one synthetic decision before reporting ready;
a native abort during load/decode fails startup with the device name and a pointer to the log.
Health includes each worker's `startup_inference_ms`. This check is outside player requests,
the label cache and completed-request counters; it proves a short inference works, not full
load capacity. The synthetic warmup uses the existing tokenizer, prompt and scoring path.
Validation: 202 tests passed, three optional native tests skipped. A real cached GGUF CPU
smoke check also classified one chat and one gameplay prompt correctly with the new logger.
These local checks are separate from the successful Titan live validation recorded above.

Multi-GPU live check (2026-10-14): both upgraded nodes initially reported two healthy workers
and empty classifier caches. The RX 6800 XT/Vulkan node completed native work on both cards
(30/32 decisions after burst plus accuracy) with zero failures, retaining 95% held-out and
100% mixed accuracy. The GTX TITAN X/CUDA node lost both worker processes on their first
inference (zero completions, one failure each); subsequent classification used wiki/social
fallback only, scoring 85/96 overall. This is a failed CUDA deployment check, not a measured
model accuracy regression; native error logs are needed to identify the cause. The pooled
100-chat burst returned 100 generated replies, median 9.943 s and p95 31.209 s, but 45 CUDA-node
requests used wiki classification fallback. Vulkan classification-phase median improved from
4.891 to 0.464 s; its full-reply median was 6.885 s, versus 18.497 s on the degraded CUDA node.
The sustained pooled simulation was withheld after the worker failures. Artifacts:
`runs/live-decider01-multigpu-accuracy-20261014.json` and the server checkout's
`artifacts/decider01-multigpu-pooled-chat-20261014.json`. Startup health/VRAM allocation alone
did not establish successful inference on the CUDA rig.

## Automatic classifier workers per GPU

With `CLASSIFIER_BACKEND=decider` and `DECIDER_GPU_LAYERS=-1`, startup now discovers the
dedicated GPUs exposed by the installed llama-cpp-python backend and loads one Decider
worker per card. `DECIDER_GPU_DEVICES=auto` is the default, including when absent from an
existing `.env`. Pull this code and restart the RAG service to apply it. CUDA and Vulkan
rigs use the same setting; keep their matching GPU wheel installed.

All workers share the existing 9092 endpoint, classifier queue and label cache. Each owns
a separate spawned process and native context, explicitly restricted to its card (no model
splitting across cards). The first idle worker takes the next request; sequential requests
rotate across idle workers. Timed-out queued work is discarded; an already executing call
keeps its worker until native inference finishes. Model/prompt/scoring, wiki routing and
the game-server endpoints stay the same. No new GGUF, network port or game-server update is needed.

Inspect names without loading model weights:

```powershell
.venv\Scripts\python.exe -m npc_rag devices
```

Optional `.env` overrides: `DECIDER_GPU_DEVICES=CUDA0` tests one NVIDIA worker;
`DECIDER_GPU_DEVICES=Vulkan0,Vulkan1` explicitly selects two Vulkan cards. Use names from
the device command; native backend numbering can differ from Windows Task Manager.
Auto excludes CPU/BLAS and integrated GPUs, deduplicating known physical device IDs when
multiple backends expose the same card. Explicit names can opt into an integrated GPU.
No visible dedicated GPU, unknown names or any worker failing to load prevents GPU-mode
startup with an error. `DECIDER_GPU_LAYERS=0` restores the original single CPU worker.

ROCm Decider workers explicitly disable flash attention when constructing their native
context. On the RX 6800 XT / gfx1030 node, the Python backend's automatic flash-attention
selection aborted during the startup decision with `GGML_ASSERT(max_blocks_per_sm > 0)`
in `fattn-common.cuh`. GPU discovery and all four student copies were already healthy.
This compatibility setting keeps Decider GPU offload and per-card pinning enabled; it
does not change `STUDENT_ARGS`, CUDA/Vulkan workers, model weights or prompt scoring.
The override is confined to context creation in each isolated worker and restored even
if creation fails. Look for `flash attention disabled for ROCm compatibility` followed
by successful worker readiness. Local regression checks cover both task types and hook
restoration; inference on the affected ROCm host must still be verified after deployment.
See the [upstream RDNA2 flash-attention report](https://github.com/ggml-org/llama.cpp/issues/24672).

`/health.classification` reports `execution=gpu-process-pool`, total `capacity`,
`healthy_capacity`, `in_flight`, `queued`, and each worker's device name, card description,
PID, busy state, completed count and failures. Two cards should show capacity 2 and
different PIDs. A crashed worker is removed from scheduling; remaining healthy workers
continue. Restart RAG to restore a failed worker. Health device assignments describe the
restricted native load target, not measured GPU utilization or offloaded-layer counts.
Workers also watch their parent process and exit if the launcher force-stops RAG, releasing
their GPU memory instead of remaining alive through a restart.

Each worker needs its own model/context VRAM and Python/Torch host memory. `DECIDER_THREADS`
is per worker. Leave room for Qwen's generation buffers on each card; insufficient memory
fails startup rather than silently reducing the configured pool. Shared GPU compute and
cached labels mean two workers do not guarantee twice the full-conversation throughput.
Keep one RAG service process per host; additional Uvicorn workers would duplicate this pool.

Device enumeration/pinning follows the pinned native ABI and llama.cpp's
[backend registry](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)
and [model device parameters](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h).
Validate locally with `.venv\Scripts\python.exe -m pytest tests/test_decider_pool.py tests/test_decider.py -q`,
then repeat live accuracy and simulated-player tests after deployment. Real multi-card GPU
placement and speed require testing on the target rigs; local process tests use model doubles.
Validation on 2026-10-14: the full suite passed 196 tests with three optional native tests skipped.
The installed 0.3.36 CPU wheel also passed the real device-listing command, correctly returning
no GPUs; CUDA/Vulkan device enumeration and model placement remain target-host checks.

Earlier single-worker GPU-configured verification (2026-10-14): both deployed nodes retained 95% held-out and
100% mixed accuracy (`runs/live-decider01-gpu-accuracy-20261014.json`). The server harness
completed 624 sustained requests with no HTTP failures; generated-reply medians at
1/2/4/8/16/24 concurrency were 1.60/1.45/2.15/3.44/5.57/1.81 seconds. The separate
100-chat burst had a 9.86-second median and 20.55-second p95, with all replies generated.
Both nodes recovered idle/healthy with zero native classifier failures. These random,
cache-warmed, overflow-enabled runs are not a controlled GPU speedup measurement, and
the five-second median is still inconsistent. Live health reports requested `gpu_layers=-1`;
the 41 ms raw benchmark was on a separate test rig. Each live service had one Decider
worker during that run; these measurements predate the automatic pool above.

## Enable Decider GPU offload on existing Windows rigs

Sakura and RAG use separate native libraries: GPU generation in Sakura does not enable GPU
classification in Python. Stop the RAG service before replacing its loaded DLLs, then run one
of these commands from the updated checkout:

```powershell
# NVIDIA CUDA rig (CUDA 12.4 wheel; suitable for Pascal-compatible builds as well):
# This prebuilt wheel omits compute capability 5.2 (GTX TITAN X); use Vulkan on those cards.
powershell -ExecutionPolicy Bypass -File ps/Install-Decider.ps1 -Backend Cuda

# Vulkan rig, including AMD with a working Vulkan graphics driver:
powershell -ExecutionPolicy Bypass -File ps/Install-Decider.ps1 -Backend Vulkan
```

Both options force-reinstall the official `llama-cpp-python==0.3.36` wheel from its backend-specific
index, bypassing cached CPU builds. They do not need a C++ compiler or Vulkan SDK. Missing wheels
fail explicitly rather than silently installing a CPU build. `-Backend Auto` (the default) retains
the existing installation behavior. After installing dependencies, the helper initializes the
native backend and verifies GPU offload support; a requested GPU backend cannot finish successfully
with offload unavailable. This verifies capability, not this model's actual placement or speed.

Set `DECIDER_GPU_LAYERS=-1` in `.env` and restart RAG with the usual launcher/task. Confirm GPU
use using native loading logs and GPU monitoring; `/health.classification.gpu_layers` reports the
requested setting only. Keep model, threads, context, routing and overflow settings identical for
the CPU/GPU A/B. GPU mode now defaults to one classifier per detected dedicated GPU; select one
device explicitly for a single-worker comparison. Revert to `DECIDER_GPU_LAYERS=0` and restart for CPU execution. The GPU rig still
needs a compatible driver/runtime; DLL errors must be resolved before benchmarking.

Official backend installation documentation: https://github.com/abetlen/llama-cpp-python#installation

## Live custom-Decider verification (2026-10-14)

Both nodes (`47.51.162.106:9092` and `192.168.1.188:9092`) now report managed release
`lidoll/decider0-1`, `decider0-1.gguf`, `game-chat-v3`, four CPU threads, zero GPU layers and
one native classification slot. Each passed the unchanged deployed hybrid accuracy suite:
**76/80 held-out (95%), 16/16 mixed (100%), 92/96 overall**; GAME 44/48 and CHAT 48/48.
This matches local qualification. Artifact: `runs/live-decider01-accuracy-20261014.json`.

A cold-classifier-cache burst sent **100 simultaneous chat requests, 50 per node**, through the
game server's existing AI pool with normal classification and configured local/OpenAI overflow.
All 100 produced guard-passed generated replies (one guard retry), all routed to general chat,
with zero HTTP failures, canned fallbacks or wiki-source replies. Median **9.122 s**, p95
**26.274 s**, max **30.826 s**, **20/100 under 5 s**: the median-under-five-seconds goal is unmet.

| Node | Requests | Response median | Response p95 | Classification median | Generation median |
|---|---:|---:|---:|---:|---:|
| 47.51.162.106 | 50 | 7.628 s | 12.112 s | 5.120 s | 1.597 s |
| 192.168.1.188 | 50 | 12.911 s | 29.238 s | 6.034 s | 10.149 s |

Phase medians do not add. Fifteen requests used wiki-based routing near the classifier deadline
(five on A, ten on B), although their replies were generated successfully. Both nodes reached
eight local and four cloud generations in flight without observed cloud cooldown. Both recovered
healthy and idle with zero native inference failures. This workload repeats ten chat prompts;
each node shared 36 pending classifier requests, so it is not 100 distinct-message throughput.
The older single-node 100-request result is not a matched comparison. No production settings
were changed during this test. Burst artifact:
`C:/Scripts/Lidollquest-server/artifacts/decider01-pooled-chat-20261014.json`.

## Hosting release: decider0-1 Q4_K_M (2026-10-14)

The selected v3 model has now been converted and tested through the native CPU Decider runtime
(`llama-cpp-python==0.3.36`, four threads, zero GPU layers). The Q4_K_M file preserves the BF16
result: **76/80 held-out (95%), 16/16 mixed (100%), 92/96 overall**, GAME recall 44/48 and CHAT
48/48. The same four GAME requests were missed and no CHAT was routed to GAME. Wiki evidence
matched the BF16 run. This is a sequential local accuracy check, not a server load/latency test.

Hosting files are in `runs/releases/decider0-1-q4km/`:

- `decider0-1.gguf` (1,274,396,192 bytes), intended URL `https://lidoll.dev/gguf/decider0-1.gguf`.
- `decider0-1-support.zip`, intended URL `https://lidoll.dev/gguf/decider0-1-support.zip`.
- `SHA256SUMS.txt` and `manifest.json` record integrity/provenance; `qualification.json` records accuracy.

Both hosted files were downloaded and verified with the startup installer. The release is
**game-chat-v3/message-first**. The large weights and tokenizer come from the site; tunable
configuration comes from the GitHub checkout, so tuning does not require rehosting the model.

On each server, update this repository with `git pull`, then set these in its existing `.env`:

```dotenv
CLASSIFIER_BACKEND=decider
DECIDER_MODEL=lidoll/decider0-1
DECIDER_GGUF_FILE=decider0-1.gguf
```

Launch normally with `powershell -ExecutionPolicy Bypass -File ps/Start-NpcRag.ps1` (or restart
the existing scheduled task). The managed alias selects the pinned GGUF even if an older
`DECIDER_GGUF_FILE` value remains in `.env`. Existing explicit Hub IDs/local paths keep their
usual behavior; change `DECIDER_MODEL` once to opt those servers into this managed release.
The Decider dependencies must already be installed in the RAG environment.

Startup verifies SHA-256 hashes in **`data/models/decider0-1/`**, downloads missing/corrupt assets,
and copies the current Git-tracked configs from **`npc_rag/decider_config/`** into that model folder.
Weights and tokenizer are fixed to the release hashes in `npc_rag/decider_release.json`.
The support ZIP supplies the tokenizer and release records; its historical config copies never
override the checkout. Downloads go through temporary files and verification before installation.
Concurrent launches share an installation lock. A failed download stops startup with a clear error;
the next launch retries. A complete cached installation works offline.

For tuning, edit **`npc_rag/decider_config/decider_config.json`** in GitHub (for example, temperature),
then `git pull` and restart RAG on both servers. `config.json`, `tokenizer_config.json`, and
`chat_template.jinja` also travel with the code; their architecture/tokenizer values must match
these weights. `routing_policy.json` records training provenance; the active question/options
remain in **`npc_rag/decider_policy.py`**, shared with training. This release requires the trained
plain, message-first layout. Fine-tuned temperature 1 remains uncalibrated; re-evaluate after tuning.
Editing the generated cache copies is temporary: startup restores the checkout's versions.

The background launcher now waits up to 20 minutes for first-time provisioning and reports early
process exits immediately. Use `-StartupTimeoutSeconds` for a different wait; the foreground
scheduled task waits for actual startup completion. Download progress is in `logs/npc-rag.log`.
Health reports `release: lidoll/decider0-1`, the local model path, and `gguf_file: decider0-1.gguf`.
Verified locally: 175 unit tests passed (3 optional native tests skipped), launcher syntax,
real hosted download, then offline native startup and GAME/CHAT smoke decisions. No live server
restart or new server latency measurement was performed.

`python/export_classifier_gguf.py` pins conversion to llama.cpp revision
`0c1e57098bba43ac29e6e3b677cdceebdd22334f`, matching the native package, and uses `--no-mtp`
because the HF config retains an unused draft-layer count while these classifier weights have
no MTP tensors. `python/package_classifier_release.py` requires a complete native evaluation
before assembling the support archive. The earlier `runs/releases/decider0-1/` directory is an
unqualified conversion intermediate with unused MTP metadata; use only the `-q4km` release above.

## Classifier training v3: diverse data and independent development (2026-10-14)

**Completed first pass:** the merged BF16 model scored **76/80 held-out (95%) and 16/16 mixed
(100%)**, passing the requested 93.8%/100% bar. With identical `hybrid-v2` routing and identical
wiki evidence, base Decider scored 87/96 and tuned Decider **92/96**. GAME recall improved from
39/48 (81.25%) to 44/48 (91.67%); CHAT remained 48/48 (100%). Missed game requests fell from
9 to 4, with zero chat-to-game errors. These final cases were not used for checkpoint selection.
Full report: [`data/classifier/v3-run-report.json`](data/classifier/v3-run-report.json).

The run completed 100 optimizer steps over two epochs in 1,258 seconds after baseline checks,
with 2.78 GiB peak PyTorch allocation. Epoch two won on development worst-class recall: 98%
GAME and 98% CHAT, versus the base's 83%/100%. Synthetic test was 99/100. The selected adapter
and merged checkpoint are under `runs/classifier-v3-first-pass/`. **No production deployment was
performed**; subsequent Q4_K_M qualification is recorded above. Fine-tuned probabilities remain uncalibrated. Ordinary
tests pass 164 cases; the three optional native tests pass separately in `.venv-train`.

The v3 GPU smoke test passed on the RTX 3080 10 GB: finite forward/backward, nonzero adapter
gradients and matching saved/reloaded predictions. Peak PyTorch allocated memory was **2.78 GiB**
(not total device memory); artifact: `runs/classifier-v3-smoke/`. The optional tiny Qwen3.5 test
also verifies the restricted A/B logits against a full forward pass. Full-run results are recorded
separately; successful training alone does not establish a routing improvement.

`data/classifier/synthetic-pilot-v2.jsonl` contains **1,000 authored messages, 500 GAME / 500 CHAT**.
It replaces the old pilot's 500 duplicate-prefix rows with new messages: short lookups, emoji,
real-world trivia, AI identity, roleplay, game-help imperatives, bugs, implicit needs and lore.
The 262 family groups include paired contrasts and larger topic families. Each family stays together:
**800 training / 100 validation / 100 test**, balanced within every split. The builder also keeps
textually similar families together, rejects normalized duplicates and excludes the fixed 96-message
evaluation suite. Source wiki and dataset hashes are recorded in the adjacent manifest. No player
logs were copied. Labels remain synthetic drafts; see `data/classifier/LABELING_GUIDE.md` for
opinion/advice, vent/problem and mixed-intent boundaries. The legacy v1 files are historical and
are no longer the trainer defaults.

`realistic-dev-v2.jsonl` adds **200 separately authored short messages**, balanced by label, with
no normalized overlap with training or either fixed evaluation file. This set chooses checkpoints
by worst-class recall, then accuracy, then loss. Synthetic validation is a secondary diagnostic.
The fixed evaluation files are exclusion checks and final qualification only.

Prefixes, punctuation, lowercasing, apostrophe removal, typos and textspeak are sampled **on the
fly each epoch with identical probabilities for both labels**. Saved epoch rows record their
`variant` operations. The `?`/no-`you` shortcut is 50% on the authored dataset; both the builder
and each full epoch reject shortcut accuracy above 65%. The manifest records preview statistics.
These checks detect a known artifact, not every possible synthetic-data shortcut.

The first runner targets an **NVIDIA RTX 3080 / 10 GB** with NF4 QLoRA, rank 8, BF16 computation,
one message per microbatch, accumulation 16, 384-token maximum prompts and two epochs. It trains
only adapters using cross-entropy on Decider's A/B option logits at the answer slot. The prompt and
described options come from the same `game-chat-v3` policy module as production; JSON contains
only the message. Training shuffles option order with corresponding target remapping, while
validation/test keep production's order. Inputs are rejected rather than silently truncated.
The full vocabulary logits are not materialized. No llama-cpp-python build is needed for training.
The base checkpoint is pinned to Mapika's v11 revision `ff99089b6810691bd79134ec6febb5d9afa6e619`.
On this Windows setup, linear-attention/causal-convolution operations use correct reference PyTorch
kernels; their optimization warnings are expected. A working smoke test does not require compiling
those optional kernels.

From this checkout in PowerShell (Python 3.11 or 3.12):

```powershell
powershell -ExecutionPolicy Bypass -File ps/Install-ClassifierTraining.ps1
powershell -ExecutionPolicy Bypass -File ps/Train-Classifier.ps1 -SmokeTest
powershell -ExecutionPolicy Bypass -File ps/Train-Classifier.ps1
```

The installer creates **`.venv-train`**, separate from RAG's `.venv`, and pins CUDA PyTorch and
training dependencies. To choose a different base Python, pass `-Python C:\path\to\python.exe`.
Stop other GPU model workloads yourself if needed before training; the scripts do not terminate
them. CUDA/PyTorch compatibility and actual peak memory must be verified by the smoke test on
each machine. The AMD cards are not configured by this first-pass runner.

The smoke test does two optimizer steps on eight training rows, verifies finite loss/nonzero
adapter gradients, then saves and reloads the adapter and compares validation probabilities.
It downloads the original trainable checkpoint on first use (several GB) into ignored
`data/models/training`, not the Q4_K_M GGUF. Full training selects the best epoch using development
per-class recall, accuracy and loss, then evaluates the selected saved adapter once on the
synthetic test partition. The fixed 96-message gate and real player questions remain outside training.

Each run gets a fresh ignored directory under **`runs/`**. It contains the exact base revision,
package versions, dataset/development hashes and routing policy in `run.json`; baseline/epoch development and validation reports;
loss/memory records in `training-metrics.jsonl`; `best-adapter/`; and, after full training,
`final-test.json`. Completion records whether development improved over the starting checkpoint.
Existing output directories are refused. A successful process means training completed, not that
the classifier passed accuracy gates. The synthetic test report includes the existing 90%-per-class
gate; synthetic success still needs verification on reviewed real inputs.

The result is an **experimental LoRA adapter**, not a deployable GGUF. `python/merge_classifier.py`
merges into the pinned original BF16 checkpoint on CPU. The saved `decider_config.json` explicitly
uses **temperature 1, uncalibrated**; stale base-model choice calibration is not presented as tuned
calibration. Fit temperature on independent development/calibration data before confidence fusion.
Evaluate the unchanged accuracy suite, quantize and
evaluate the quantized model before replacing a RAG model. Training does not edit `.env`, start
services, upload weights or deploy the adapter. This domain-only pass has no generic-task replay;
do not assume it preserves Decider's unrelated decision skills.

After training, compare the base and merged checkpoints with the **same hybrid CLI** and wiki:

```powershell
.venv-train\Scripts\python.exe python/merge_classifier.py --run runs/classifier-v3-first-pass
$BaseCheckpoint = 'data/models/training/models--Mapika--decider-2b/snapshots/ff99089b6810691bd79134ec6febb5d9afa6e619'
.venv-train\Scripts\python.exe -m npc_rag eval --classifier-backend decider --include-mixed --decider-torch-model $BaseCheckpoint --evidence-url http://192.168.1.188:9092 --output runs/classifier-v3-first-pass/base-heldout.json
.venv-train\Scripts\python.exe -m npc_rag eval --classifier-backend decider --include-mixed --decider-torch-model runs/classifier-v3-first-pass/merged --evidence-url http://192.168.1.188:9092 --output runs/classifier-v3-first-pass/tuned-heldout.json
.venv-train\Scripts\python.exe python/compare_classifier_runs.py --run runs/classifier-v3-first-pass --base runs/classifier-v3-first-pass/base-heldout.json --tuned runs/classifier-v3-first-pass/tuned-heldout.json --output data/classifier/v3-run-report.json
```

`--decider-torch-model` is evaluation-only, uses CUDA/BF16 and does not change server configuration.
The evaluator returns exit code 1 when either class misses the 90% gate; it still saves the result.
The comparison rejects different policies, cases, thresholds or wiki scores. It reports both class
recalls and both error directions. A higher synthetic score is never reported as final improvement.
These BF16 results do not qualify a future Q4_K_M export or establish live response latency.

Decider labels now retain `game_probability` through the in-memory cache and expose it in verdict
scores when the model runs. Existing string-label callers remain compatible. The existing wiki
override thresholds are unchanged; wiki-first decisions need not run Decider and have no model
probability. Previous-NPC context remains deferred: input and cache still use the message alone.

The requested Qwen-4B comparison bar is **75/80 held-out (93.8% rounded) and 16/16 mixed (100%)**.
`compare_classifier_runs.py` reports that target separately from improvement over base Decider
and the existing per-class 90% gate. Neither final subset selects the training checkpoint.

### Planned CPU optimization: fixed prefix first

The next controlled format is `game-chat-v4-schema-first`: **same wording**, question/options
before the JSON message. It is supported explicitly by both the shared training encoder and
the inference adapter, including option-order remapping. Checkpoint metadata records
`routing_prompt_order`, `routing_policy` and `schema_first_trained`; policy/order mismatches fail.
The existing v3 experiment remains message-first. To start a separate prefix-first experiment:

The separate v4 two-step GPU smoke test also passed finite gradients and save/reload agreement
(`runs/classifier-v4-prefix-smoke/`, 2.78 GiB peak allocation). This validates the new training path;
no full v4 run, final accuracy result or cached CPU benchmark is claimed.

```powershell
powershell -ExecutionPolicy Bypass -File ps/Train-Classifier.ps1 -PromptLayout schema_first -SmokeTest
powershell -ExecutionPolicy Bypass -File ps/Train-Classifier.ps1 -PromptLayout schema_first
```

**This prepares the layout, not the speedup.** Installed `decider-ai==1.9.0` clears llama.cpp memory
at every `GGUFEngine._decode`; its `score_shared` calls the full scorer. Its `schema()` cache is
Torch-only, and setting `schema_first` alone does not alter `decide()` token ordering. Our explicit
encoder supplies the new order, but GGUF still recomputes it and health reports `prefix_cache: false`.
With the pinned tokenizer, all 200 development inputs share a **74-token fixed prefix**; their
remaining suffixes range from 16 to 22 tokens (mean 18.06). These are token counts, not timing results.
The next native optimization must preserve and restore the full fixed-prefix state (including
Qwen3.5 recurrent state), decode only the suffix, and demonstrate identical probabilities to
uncached scoring on alternating/repeated messages before timing warm/cold CPU latency. Retain
single-worker ownership and cancellation guarantees. Do not remove context clearing without
that state-isolation test. Shortening options is a later separate experiment after accuracy passes;
it must retain lore, place, gameplay-problem and off-topic distinctions.

Data-only validation works without GPU dependencies:

```powershell
.venv\Scripts\python.exe python/train_classifier.py --check-data
# Regenerate the same grouped split from the authored sources:
.venv\Scripts\python.exe python/build_classifier_dataset.py --wiki-root C:\Users\langley\GameMakerProjects\lidollquest\web\wiki
```

## Synthetic classifier seed (2026-10-14)

`data/classifier/synthetic-seed-v1.jsonl` contains **120 draft examples**, balanced 60 `game` /
60 `general`, in 60 contrasting pairs across the 12 player-wiki topics. Both labels use similar
subject vocabulary: requests for rules or game facts are GAME; social remarks, personal preferences,
stories and reassurance are CHAT. Mixed emotional/greeting-plus-gameplay requests remain GAME.
These are assistant-authored messages, not copied player conversations or generated NPC answers.
Labels are proposed and every row is `needs_review`; no training or deployment has been performed.

Regenerate the seed and provenance manifest from the checked-in authoring script:

```powershell
.venv\Scripts\python.exe python/build_classifier_seed.py --wiki-root C:\Users\langley\GameMakerProjects\lidollquest\web\wiki
```

The builder checks message limits, normalized duplicates and exact normalized overlap with both
fixed evaluation files; it records hashes of those files, source wiki pages and the output dataset.
This is not a semantic leakage check or independent label review. Metadata (`label`, `family_id`,
`topic`, source page, review status) must stay outside model inputs. Keep each family and any derived
paraphrases together in future data splits, and review semantic duplicates across families too.
Train the same message/context representation that the service will actually supply.

Suggested experiment, not a proven data requirement: review this seed, grow to 500-1,000 varied
training examples, then compare 500/1,000/2,000-example runs against the same development set.
Budget 2,000-5,000 reviewed training examples if more data continues to help. Start balanced by
class, but include ordinary chat as well as these game-topic social contrasts. Add slang, typos,
fragments and mixed intents across both classes; random greeting prefixes are not new independent
examples. Reserve an additional 200-500 reviewed examples for development/final evaluation, split
by conversation and paraphrase family; supplement with real player inputs as they become available.
The 43 confirmed NPC inputs in the current database export are useful but too few for a confident
standalone release estimate. Existing model-produced labels need correction, not automatic reuse.
Ambiguous fragments requiring absent conversation context should be flagged for review instead of
assigned an arbitrary label. Keep the existing 96-message gate out of all training and report
GAME recall and CHAT recall separately. Check accuracy again after GGUF quantization.

## Optional local Decider 2B Q4_K_M classifier (2026-10-14)

Reported v2 raw evaluation (October 14, approximately 12:00 Pacific): 81/96 correct;
GAME 33/48 (68.75%), CHAT 48/48 (100%), all 96 via `raw_model`. This fails the 90%-per-class
gate. The console contains no timings or policy metadata; the full artifact remains on the host
at `artifacts/decider-raw-v2.json`. These raw results must not be confused with the earlier v1
hybrid test, which happened to have identical totals. Full v2 hybrid accuracy is still unverified.

The current adapter supplies only JSON message text, without explicit RPG setting. To test whether
that missing context contributes to ambiguous decisions, a diagnostic-only comparison adds a short
game setting while keeping the production question/options unchanged. It uses 16 separate paired
factual/social development cases (32 decisions), alternates variant order, and reports per-class
counts, probabilities and median latency. It does not change production or qualify model accuracy:

```powershell
.venv\Scripts\python.exe python/diagnose_decider.py --game-context --output artifacts/decider-context.json
```

Avoid concurrent load tests: this command loads another native model copy. Its request construction
was smoke-tested locally with a model double; real context-comparison results remain pending.

Diagnostic follow-up (October 14): the host's 35 development probes passed all five controls,
all five reversed controls and the repeated control (identical probabilities). The v1 prompt
scored 5/6 gameplay/social cases, missing a greeting followed by a keyboard-shortcut question;
reversing the options did not fix it. A short question with descriptive options scored 6/6 with
JSON state, reducing the six-case median from 1,387 ms to 813.5 ms (41%). Plain text also scored
6/6 at 750 ms. These small sequential probes suggest prompt sensitivity, not a universally stuck
CHAT readout; strong wiki matches hid other model decisions in the earlier hybrid test below.
They do not qualify accuracy or establish conversation latency under load.

The prepared `game-chat-v2` policy uses the exact tested short question/descriptive options and
retains JSON message encoding. It maps those descriptions back to the existing `game`/`general`
API categories. No dependency rebuild is needed for this prompt change. Update the server checkout,
run the unchanged 96-message raw and hybrid evaluations below, then restart 9092 and confirm
`/health.classification.decision_policy=game-chat-v2`. The raw v2 gate failed as recorded above;
live deployment/hybrid results remain unverified. Require 90% in each class before another burst.

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

To investigate the all-CHAT result on the affected host, run:

```powershell
.venv\Scripts\python.exe python/diagnose_decider.py
```

This uses the same configured GGUF loader in a separate process, without modifying the running
service or its cache. It temporarily loads another model copy; avoid overlapping it with load tests.
It records runtime versions, calibration, native/Hugging Face option-letter mappings, a synthetic
prompt and answer slots, and raw probabilities for 35 development probes. These compare basic
non-game controls, reversed choices, the frozen v1 question (`legacy_v1` groups), shorter described options, plain text
versus JSON state, and a repeated control. No held-out evaluation examples are used. Output goes to
`artifacts/decider-diagnostic.json`; the console prints individual probes and group summaries.
Correct controls with failed production routing suggest a task/prompt issue; consistently wrong or
position-dependent controls point toward a wider runtime/readout/model issue. These are diagnostic
signals, not conclusive root causes. The existing HTTP classification endpoint hides probabilities
and may cancel native waits when wiki evidence wins, so it cannot perform this comparison alone.

`CLASSIFIER_BACKEND=decider` now defaults to the managed `lidoll/decider0-1` release
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
DECIDER_MODEL=lidoll/decider0-1
DECIDER_GGUF_FILE=decider0-1.gguf
DECIDER_THREADS=4
DECIDER_GPU_LAYERS=0
DECIDER_CONTEXT=4096
```

`DECIDER_MODEL` also accepts a local directory containing that GGUF, the matching Hugging Face
tokenizer files, and `decider_config.json`. A GGUF alone is insufficient. Relative directories
resolve against this checkout. With a Hub ID, the library downloads required files into the usual
Hugging Face cache on first load; configure `HF_HOME` before launching if a different cache is needed.
The launcher allows 20 minutes by default for first download/load. Startup fails clearly if
dependencies/config/model cannot load. To explicitly use the original upstream checkpoint, set
`DECIDER_MODEL=Mapika/decider-2b-GGUF` and `DECIDER_GGUF_FILE=decider-2b-v11-Q4_K_M.gguf`.

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
`/health.classification.backend` should be `decider`, with `decision_policy=game-chat-v2`, model/file,
CPU threads, configured GPU layers, context, capacity, busy state and failure count. `classifier_url`
is null because no HTTP classifier is used. Existing `method=llm` and `scores.llm` retain their meanings
as a model-based category, including Decider, for client compatibility. Revert to
`CLASSIFIER_BACKEND=llamacpp` and restart to use the previous CLASSIFIER_URL again.

One worker owns one native context. CPU mode uses one worker; GPU mode defaults to one spawned
worker per detected dedicated GPU (see automatic workers above). `DECIDER_THREADS` controls CPU
threads per worker (1-64), not parallel model copies. `DECIDER_GPU_LAYERS=0` selects CPU; -1 requests all layers on a GPU with
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
Validation: 144 local tests pass, including descriptive-option mapping, rejection of stale terse labels,
native cancellation/recovery and both evaluation modes. These use a model double, not native weights.
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

## Ambient NPC lines are never billed (2026-10-14)

The game's Ambient Life (`AMBIENT_NPC_LIFE_PLAN.md` in the game repo) sends new `/v1/companion/chat` kinds:
- `ambient_open`, `ambient_reply` and `ambient_chime`: one NPC to another;
- `ambient_greet` and `ambient_answer`: an NPC to a player;
- `shop_beckon`: a shopkeeper to a passer-by.

They use the resident prompt (`RESIDENT_STATIC`: someone who lives here, not a companion), plus optional `thread` lines (the conversation so far).

**Cost guard**
- **Only `PAID_KINDS` can overflow.** `PAID_KINDS` in `companion/prompts.py` is `chat`, `emote`, `reaction`, `musing` and `banter`: the companions' kinds. Only these may reach the paid model. Pip's `/v1/npc/chat` also keeps its overflow. Every other kind runs on the local model only, and any kind added later is local unless added to `PAID_KINDS` on purpose.
- **Ambient kinds leave room for players.** They use `background()`, which needs more than `AMBIENT_LOCAL_RESERVE` free local slots (default 2, clamped below capacity). Otherwise the node answers **429 `local_busy`** at once, without queueing. The game server skips the line, and a 4xx never marks the node down.
- **Every reply reports `lane`.** It is `local` or `cloud`. An ambient line that somehow came back `cloud` is logged as `COST GUARD` here and dropped by the game server.

Optional `.env` setting:

    AMBIENT_LOCAL_RESERVE=2

Ambient replies also need the participation model on (`PARTICIPATION_ENABLED=true`). The game server uses it to decide who speaks. Without it, the game keeps ambient life silent.

Tests: `tests/test_ambient_cost_guard.py`.

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

**Keyword evidence (classifier policy `hybrid-v2`, 2026-10-14).** Short keyword questions ("whats a dive") embed weakly, so a second rule also makes a message a game question: best wiki score at least `RAG_KEYWORD_DENSE` (0.62) **and** keyword coverage at least `RAG_KEYWORD_COVERAGE` (0.5). Coverage is the share of the message's IDF-weighted words found in the single best BM25 section; words the wiki never uses count at full weight, so off-topic questions stay low. Thresholds were picked on the 500 authored rows of `data/classifier/synthetic-pilot-v1.jsonl` (game recall 72.4% → 76.0%, chat forced to game unchanged at 1.6%), not on the eval files. Wiki-only held-out + mixed accuracy: 84.4% → 86.5% (game 68.8% → 72.9%, chat still 100%). `/v1/search` now returns `keyword_coverage`; set `RAG_KEYWORD_COVERAGE=2` to turn the rule off.

**Routing scores in the logs.** Pip and companion log lines now carry the classifier signals in brackets, e.g. `game/wiki passed 812ms [llm=general wiki_best=0.6431 wiki_keyword=0.71]: 'whats a dive'`. Lines where `llm=general` but the category is `game` (or `llm=game` with a low `wiki_best`) are model/wiki disagreements — the best real messages to hand-label for classifier training.

**Wiki first for the Decider (2026-10-14).** A native Decider run cannot be stopped once started, so with `CLASSIFIER_BACKEND=decider` the wiki check (~70 ms) now runs first and the Decider only runs on weak evidence. Live on both nodes, 39% of eval messages were decided by the wiki; each one previously still ran a full Decider pass in the background, and the next message waited behind it (p50 ~1.9 s instead of ~1.0 s). HTTP backends keep the parallel start, since llama-server drops abandoned requests.

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

## Optional written speaking style

Both `/v1/companion/chat` and `/v1/npc/chat` accept a top-level `speaking_style` field with `normal`, `gentle` or `baby_talk`. For example, add `"speaking_style":"baby_talk"` alongside the usual message, NPC identity and persona fields. Omission or null preserves existing persona behavior; an invalid value returns 422. The existing `voice` array still holds dialogue examples.

The runtime serializes only a compact JSON control marker into the system prompt, not a long style instruction. Fine-tune the student with that same marker to teach reliable switching. The selector is not zero-token metadata or a speech-synthesis setting. Installing this code does not train a model or change existing runtime configuration.
