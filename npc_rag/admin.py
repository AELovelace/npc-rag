"""Control page for one npc-rag node: python -m npc_rag admin (ADMIN_PORT, default 9091).

Node port layout (Linux and Windows): 9090 kept free (old Sakura copies, or Cockpit if installed), 9091 this page, 9092 the npc-rag API,
9100+ the student llama-server copies (127.0.0.1 only).
The page runs as its own process (npc-rag-admin.service) so it stays up while npc-rag restarts, and it can:
  * show health: npc-rag /health, the systemd unit, GPUs, memory, disk and the journal tail;
  * edit a whitelisted set of .env keys (secrets are write-only), validated by load_settings in a clean subprocess,
    written atomically with a timestamped backup;
  * restart npc-rag (systemctl --no-block restart, allowed by the polkit rule in linux/) and run a speed sweep on the
    local copies.
Login is ADMIN_PASSWORD only (never NPC_API_KEY); every admin action is appended to data/admin-audit.jsonl without values.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .config import ROOT, Settings

PAGE = Path(__file__).with_name("admin_page.html")  # The single-page UI (plain HTML + JS, no CDN).
COOKIE = "npc_admin"                                 # Signed session cookie name.
SESSION_S = 12 * 3600                                # A login lasts 12 hours (or until this page restarts).
FAILS, FAIL_WINDOW_S = 5, 300                        # Five wrong passwords per IP per 5 minutes, then 429.
BACKUPS = 5                                          # .env backups kept beside the file.
LINUX = sys.platform.startswith("linux")
CSRF_HEADER = "x-npc-admin"                          # Every POST needs this header: a cross-site form can't send it.

# key: (group, kind, help). kind: bool | int | text | secret | choice:a|b. Anything not listed here can't be edited.
FIELDS: dict[str, tuple[str, str, str]] = {
    "STUDENT_FLEET": ("Student fleet", "bool", "npc-rag downloads the pinned lidoll-fm GGUF and runs its own llama-server copies."),
    "STUDENT_MODEL": ("Student fleet", "text", "Empty = the release list's default; e.g. lidoll-fm-0.3 rolls back."),
    "LLAMA_SERVER": ("Student fleet", "text", "Path to llama-server; empty = PATH or /opt/llama.cpp/*/bin."),
    "STUDENT_DEVICES": ("Student fleet", "text", "auto, or a list such as ROCm0,ROCm1 / Vulkan0,Vulkan1 / CUDA0."),
    "STUDENT_COPIES_PER_DEVICE": ("Student fleet", "int", "llama-server copies per GPU."),
    "STUDENT_SLOTS": ("Student fleet", "int", "-np per copy: replies each copy writes at once."),
    "STUDENT_CTX_PER_SLOT": ("Student fleet", "int", "Context tokens per slot (8192 fits a companion prompt)."),
    "STUDENT_BASE_PORT": ("Student fleet", "int", "First copy port (127.0.0.1); 9090-9092 are always skipped."),
    "STUDENT_ARGS": ("Student fleet", "text", "Extra llama-server flags for every copy."),
    "LLM_URL": ("Models", "text", "Main model URL(s), comma list. Ignored when STUDENT_FLEET is on."),
    "LLM_CONCURRENCY": ("Models", "int", "Replies per LLM_URL copy at once. Ignored when STUDENT_FLEET is on."),
    "LLM_MAX_TOKENS": ("Models", "int", "Pip reply token cap."),
    "COMPANION_MAX_TOKENS": ("Models", "int", "Companion reply token cap."),
    "LLM_TIMEOUT": ("Models", "text", "Seconds before a reply gives up."),
    "CLASSIFIER_BACKEND": ("Decider", "choice:llamacpp|decider", "decider = the native Decider worker (no classifier port)."),
    "CLASSIFIER_URL": ("Decider", "text", "Only for CLASSIFIER_BACKEND=llamacpp."),
    "DECIDER_GPU_LAYERS": ("Decider", "int", "0 = CPU worker."),
    "DECIDER_GPU_DEVICES": ("Decider", "text", "auto, or a device list."),
    "DECIDER_THREADS": ("Decider", "int", "CPU threads per Decider worker."),
    "PARTICIPATION_ENABLED": ("Decider", "bool", "JOIN/QUIET answers for ambient chat (needs the decider backend)."),
    "OPENAI_API_KEY": ("Overflow", "secret", "Non-empty enables paid overflow for player replies."),
    "OPENAI_MODEL": ("Overflow", "text", "Overflow model id."),
    "OPENAI_LOCAL_CAPACITY": ("Overflow", "int", "Local replies in flight before overflow starts."),
    "OPENAI_CONCURRENCY": ("Overflow", "int", "Overflow replies at once."),
    "NPC_API_KEY": ("Secrets", "secret", "Shared key the game server sends; change both sides together."),
    "LLM_API_KEY": ("Secrets", "secret", "Only if llama-server runs with --api-key."),
}
_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")  # KEY=value, the same lines config._load_dotenv reads.


def env_path(env_file: str | None = None) -> Path:
    """The .env file npc-rag reads (same rule as config.load_settings)."""
    chosen = Path(env_file or os.environ.get("NPC_ENV_FILE", ".env"))
    return chosen if chosen.is_absolute() else ROOT / chosen


def read_env(path: Path) -> dict[str, str]:
    """KEY -> value from the file (quotes stripped like the loader does); comments and blanks skipped."""
    out: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip().startswith("#"):
                continue  # Commented-out examples are not settings.
            m = _KEY_LINE.match(raw)
            if m:
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def public_settings(path: Path) -> list[dict]:
    """The editable keys with their current file values; secrets only say whether they are set."""
    values = read_env(path)
    rows = []
    for key, (group, kind, help_) in FIELDS.items():
        row = {"key": key, "group": group, "kind": kind, "help": help_}
        if kind == "secret":
            row["set"] = bool(values.get(key))  # Write-only: the value never leaves the machine through this page.
        else:
            row["value"] = values.get(key, "")
        rows.append(row)
    return rows


def check_changes(changes: dict) -> dict[str, str]:
    """Only whitelisted keys, single-line string values; returns the cleaned dict or raises ValueError."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("nothing to change")
    clean = {}
    for key, value in changes.items():
        if key not in FIELDS:
            raise ValueError(f"{key} can't be edited here")
        if not isinstance(value, str) or len(value) > 2000 or any(c in value for c in "\r\n\0"):
            raise ValueError(f"{key} must be one line of text")
        kind = FIELDS[key][1]
        value = value.strip()
        if kind == "bool" and value.lower() not in ("on", "off", "true", "false", "1", "0"):
            raise ValueError(f"{key} must be on or off")
        if kind == "int" and not re.fullmatch(r"\d{1,6}", value):
            raise ValueError(f"{key} must be a whole number")
        if kind.startswith("choice:") and value not in kind[7:].split("|"):
            raise ValueError(f"{key} must be one of {kind[7:].replace('|', ', ')}")
        clean[key] = value
    return clean


def render_env(text: str, changes: dict[str, str]) -> str:
    """The file with each changed KEY= line replaced in place; keys the file lacks are appended at the end."""
    lines, seen = text.splitlines(), set()
    for i, raw in enumerate(lines):
        m = None if raw.strip().startswith("#") else _KEY_LINE.match(raw)
        if m and m.group(1) in changes:
            lines[i] = f"{m.group(1)}={changes[m.group(1)]}"  # Comments stay on their own lines (the loader keeps inline ones).
            seen.add(m.group(1))
    extra = [f"{k}={v}" for k, v in changes.items() if k not in seen]
    if extra:
        lines += ["", "# -- Set from the npc-rag control page"] + extra
    return "\n".join(lines) + "\n"


def validate_env_file(candidate: Path, *also: Path) -> str | None:
    """Run load_settings on the candidate in a clean subprocess; None = valid, else its error message."""
    drop = set(read_env(candidate))
    for p in also:
        drop |= set(read_env(p))  # This process already copied the old .env into os.environ; don't let it mask the new file.
    env = {k: v for k, v in os.environ.items() if k not in drop and k != "NPC_ENV_FILE"}
    code = "import sys; from npc_rag.config import load_settings; load_settings(sys.argv[1])"
    try:
        done = subprocess.run([sys.executable, "-c", code, str(candidate)], cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "settings check timed out"
    if done.returncode == 0:
        return None
    lines = [ln for ln in (done.stderr or done.stdout).strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else f"settings check failed ({done.returncode})"  # The ValueError line.


def write_env(path: Path, changes: dict[str, str]) -> Path | None:
    """Validate, back up and atomically replace the .env file. Returns the backup path (None for a new file)."""
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    new = render_env(old, changes)
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        if os.name == "posix":
            os.chmod(tmp, 0o600)  # The file holds keys: owner only.
        error = validate_env_file(tmp, path)
        if error:
            raise ValueError(error)
        backup = None
        if path.is_file():
            backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(path, backup)
            for stale in sorted(path.parent.glob(f"{path.name}.bak-*"))[:-BACKUPS]:
                stale.unlink(missing_ok=True)  # Keep only the newest few backups.
        os.replace(tmp, path)  # Atomic on the same filesystem: npc-rag never reads half a file.
        return backup
    finally:
        tmp.unlink(missing_ok=True)


def _run(cmd: list[str], timeout: float = 10) -> tuple[int, str]:
    """A local command's exit code and combined output; (127, reason) when it can't run."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, str(error)


def service_state(unit: str) -> dict:
    """systemd's view of the npc-rag unit (Linux only)."""
    if not LINUX or not shutil.which("systemctl"):
        return {"available": False}
    code, out = _run(["systemctl", "show", unit, "--property=ActiveState,SubState,MainPID,NRestarts,"
                      "ActiveEnterTimestamp,MemoryCurrent"])
    info = dict(line.split("=", 1) for line in out.splitlines() if "=" in line) if code == 0 else {"error": out[:200]}
    return {"available": True, "unit": unit, **info}


def update_status() -> dict:
    """Persistent job result plus systemd state, including interrupted or killed updates."""
    result = {"available": False, "running": False, "phase": "idle", "message": "Repository updates require the Linux updater service."}
    if not LINUX or not shutil.which("systemctl"):
        return result
    code, out = _run(["systemctl", "show", "npc-rag-update.service", "--property=LoadState,ActiveState,SubState,Result"], timeout=5)
    unit = dict(line.split("=", 1) for line in out.splitlines() if "=" in line) if code == 0 else {}
    result["available"] = unit.get("LoadState") == "loaded" and (ROOT / "python/update_repo.py").is_file()
    if result["available"]:
        result["message"] = "Check the tracked Git branch and install updates when available."
    try:
        saved = json.loads((ROOT / "data/repo-update.json").read_text(encoding="utf-8"))
        result.update({k: saved[k] for k in ("phase", "message", "branch", "current", "target", "installed", "behind", "started_utc") if k in saved})
    except (OSError, ValueError, TypeError):
        pass  # Only known summary fields leave this endpoint, never repository config or command output.
    result["running"] = unit.get("ActiveState") in ("activating", "active", "reloading", "deactivating")
    if not result["running"] and result["phase"] in ("checking", "installing", "dependencies", "restarting", "restarting_admin"):
        result.update(phase="interrupted", message="The updater stopped before completion. Inspect journalctl -u npc-rag-update.service.")
    elif unit.get("ActiveState") == "failed" and result["phase"] != "failed":
        result.update(phase="failed", message="The updater service failed. Inspect journalctl -u npc-rag-update.service.")
    return result


def gpu_state() -> dict:
    """Best-effort GPU table from rocm-smi or nvidia-smi; {} when neither is installed."""
    if shutil.which("rocm-smi"):
        code, out = _run(["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showproductname", "--json"])
        if code == 0:
            try:
                return {"tool": "rocm-smi", "cards": json.loads(out)}
            except ValueError:
                pass
    if shutil.which("nvidia-smi"):
        code, out = _run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                          "--format=csv,noheader,nounits"])
        if code == 0:
            cols = ("name", "util_pct", "mem_used_mb", "mem_total_mb", "temp_c")
            return {"tool": "nvidia-smi", "cards": [dict(zip(cols, (v.strip() for v in row.split(","))))
                                                    for row in out.splitlines() if row.strip()]}
    return {}


def host_state() -> dict:
    """Load, memory and disk of this machine."""
    info: dict = {"platform": sys.platform}
    disk = shutil.disk_usage(ROOT)
    info["disk_free_gb"] = round(disk.free / 2**30, 1)
    if hasattr(os, "getloadavg"):
        info["load"] = [round(x, 2) for x in os.getloadavg()]
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        rows = dict(line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line)
        kb = lambda k: int(rows.get(k, "0 kB").split()[0])  # noqa: E731 - meminfo values are "123 kB".
        info["mem_total_gb"], info["mem_free_gb"] = round(kb("MemTotal") / 2**20, 1), round(kb("MemAvailable") / 2**20, 1)
    return info


def journal(unit: str, lines: int) -> str:
    """The unit's last journal lines (Linux)."""
    if not LINUX or not shutil.which("journalctl"):
        return "The journal is only available on Linux (journalctl)."
    return _run(["journalctl", "-u", unit, "-n", str(lines), "--no-pager", "-o", "short-iso"], timeout=15)[1]


def restart_command(s: Settings) -> list[str] | None:
    """ADMIN_RESTART_CMD, else systemctl on Linux; None = no way to restart from here."""
    if s.admin_restart_cmd:
        return shlex.split(s.admin_restart_cmd, posix=os.name == "posix")
    if LINUX:
        return ["systemctl", "--no-ask-password", "--no-block", "restart", s.admin_service]  # --no-block: the fleet may take minutes to load.
    return None


def _scrub(value):
    """Drop anything that looks like a credential from /health before it reaches the page."""
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not re.search(r"key|token|secret|password|auth", k, re.I)}
    return value


SWEEP_PROMPT = ("You are Padma, a cheerful stall keeper in the Grand Bazaar. A traveller asks what you sell today. "
                "Answer in two friendly sentences.")


async def sweep_copy(client: httpx.AsyncClient, url: str, level: int, tokens: int = 96) -> dict:
    """`level` replies at once on one llama-server copy: total tokens per second and seconds per reply."""
    async def one() -> tuple[int, float]:
        start = time.perf_counter()
        r = await client.post(url + "/completion", json={"prompt": SWEEP_PROMPT, "n_predict": tokens,
                                                          "temperature": 0.7, "cache_prompt": True})
        r.raise_for_status()
        return int(r.json().get("tokens_predicted") or 0), time.perf_counter() - start
    start = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(level)))
    wall = time.perf_counter() - start
    produced = sum(n for n, _ in results)
    return {"url": url, "level": level, "tok_s": round(produced / wall, 1) if wall else None,
            "reply_s": round(sum(s for _, s in results) / len(results), 2)}


def create_admin_app(s: Settings, env_file: str | None = None, audit_file: Path | None = None) -> FastAPI:
    """The control page app. Refuses to build without ADMIN_PASSWORD (8+ characters)."""
    if len(s.admin_password) < 8:
        raise ValueError("ADMIN_PASSWORD must be set (8+ characters) before the control page can start")
    app = FastAPI(title="npc-rag control", docs_url=None, redoc_url=None, openapi_url=None)  # No public API docs.
    secret = secrets.token_bytes(32)  # Per process: restarting the page signs everyone out.
    fails: dict[str, deque] = defaultdict(deque)
    env = env_path(env_file)
    audit_path = audit_file or ROOT / "data" / "admin-audit.jsonl"
    rag = f"http://{'127.0.0.1' if s.host in ('', '0.0.0.0', '::') else s.host}:{s.port}"  # npc-rag on this machine.
    sweep: dict = {"running": False, "results": [], "error": None, "at": None}
    tasks: set = set()  # Holds the running sweep task so it isn't garbage-collected mid-run.
    update_gate = asyncio.Lock()
    update_accepted_until = 0.0  # Cover the short gap between systemd accepting the job and activating its unit.

    async def repository_state():
        view = await asyncio.to_thread(update_status)
        if s.admin_service != "npc-rag.service" or s.port != 9092:
            view.update(available=False, message="The updater unit supports npc-rag.service on port 9092; use the CLI for custom deployments.")
        if time.monotonic() < update_accepted_until:
            view["running"] = True
        return view

    async def no_update():
        if update_gate.locked() or (await repository_state())["running"]:
            raise HTTPException(409, "A repository update is running; wait before changing settings, restarting or sweeping.")

    def audit(request: Request, action: str, **extra) -> None:
        """One line per admin action: when, from where, what (never values)."""
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ip": request.client.host if request.client else "", "action": action, **extra}
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def sign(expires: int) -> str:
        return f"{expires}.{hmac.new(secret, str(expires).encode(), hashlib.sha256).hexdigest()}"

    def signed_in(request: Request) -> bool:
        token = request.cookies.get(COOKIE, "")
        expires, _, _ = token.partition(".")
        return expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(token, sign(int(expires)))

    def require(request: Request, post: bool = False) -> None:
        """401 without a valid session; POSTs must also carry the page's own header (CSRF)."""
        if not signed_in(request):
            raise HTTPException(401, "sign in first")
        if post and request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(403, "missing control page header")

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Options"] = "DENY"  # Never inside someone else's frame (clickjacking).
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def page():
        return PAGE.read_text(encoding="utf-8")

    @app.post("/api/login")
    async def login(request: Request):
        ip = request.client.host if request.client else ""
        cutoff = time.time() - FAIL_WINDOW_S
        for stale in [k for k, v in fails.items() if not v or v[-1] < cutoff]:
            del fails[stale]  # Addresses with no recent failures drop out of the table.
        recent = fails[ip]
        while recent and recent[0] < cutoff:
            recent.popleft()  # Forget failures older than the window.
        if len(recent) >= FAILS:
            raise HTTPException(429, "too many wrong passwords; wait a few minutes")
        if request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(403, "missing control page header")
        body = await request.json()
        password = str(body.get("password", "")) if isinstance(body, dict) else ""
        if not hmac.compare_digest(password.encode(), s.admin_password.encode()):
            recent.append(time.time())
            audit(request, "login_failed")
            await asyncio.sleep(0.5)  # Slows guessing a little more.
            raise HTTPException(401, "wrong password")
        recent.clear()
        audit(request, "login")
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, sign(int(time.time()) + SESSION_S), max_age=SESSION_S, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return response

    @app.post("/api/logout")
    async def logout(request: Request):
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/status")
    async def status(request: Request):
        require(request)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(rag + "/health", headers={"X-Api-Key": s.api_key} if s.api_key else {})
                health = _scrub(r.json())
        except Exception as error:  # npc-rag restarting or down: the page still works.
            health = {"error": f"npc-rag not answering on {s.port}: {type(error).__name__}"}
        service, gpus, host = await asyncio.gather(asyncio.to_thread(service_state, s.admin_service),
                                                   asyncio.to_thread(gpu_state), asyncio.to_thread(host_state))
        return {"health": health, "service": service, "gpus": gpus, "host": host, "sweep": sweep, "update": await repository_state(),
                "restart_available": restart_command(s) is not None, "env_file": str(env)}

    @app.get("/api/update")
    async def repository_status(request: Request):
        require(request)
        return await repository_state()

    @app.post("/api/update")
    async def repository_update(request: Request):
        nonlocal update_accepted_until
        require(request, post=True)
        if update_gate.locked():
            raise HTTPException(409, "An update request is already being started")
        async with update_gate:
            view = await repository_state()
            if not view["available"]:
                raise HTTPException(400, view["message"])
            if view["running"] or sweep["running"]:
                raise HTTPException(409, "Wait for the current update or speed sweep to finish")
            code, _ = await asyncio.to_thread(_run, ["systemctl", "--no-ask-password", "--no-block", "start", "npc-rag-update.service"])
            audit(request, "repository_update", code=code)
            if code:
                raise HTTPException(502, "Could not start the updater. Install its systemd unit and polkit rule with linux/install-ubuntu.sh.")
            update_accepted_until = time.monotonic() + 10
            return JSONResponse({"ok": True, "message": "Checking the tracked branch; available updates will be installed."}, status_code=202)

    @app.get("/api/logs")
    async def logs(request: Request, lines: int = 200):
        require(request)
        return {"text": await asyncio.to_thread(journal, s.admin_service, max(20, min(1000, lines)))}

    @app.get("/api/settings")
    async def get_settings(request: Request):
        require(request)
        return {"env_file": str(env), "fields": public_settings(env)}

    @app.post("/api/settings")
    async def set_settings(request: Request):
        require(request, post=True)
        await no_update()
        body = await request.json()
        try:
            changes = check_changes(body.get("changes") if isinstance(body, dict) else None)
            backup = await asyncio.to_thread(write_env, env, changes)
        except ValueError as error:
            raise HTTPException(400, str(error))
        audit(request, "settings", keys=sorted(changes))
        return {"ok": True, "backup": backup.name if backup else None, "restart_needed": True}

    @app.post("/api/restart")
    async def restart(request: Request):
        require(request, post=True)
        await no_update()
        cmd = restart_command(s)
        if not cmd:
            raise HTTPException(400, "set ADMIN_RESTART_CMD to restart npc-rag from this page on Windows")
        code, out = await asyncio.to_thread(_run, cmd, 60)
        audit(request, "restart", code=code)
        if code:
            raise HTTPException(502, f"restart failed ({code}): {out[:300]}")
        return {"ok": True}

    @app.post("/api/sweep")
    async def start_sweep(request: Request):
        require(request, post=True)
        await no_update()
        if sweep["running"]:
            raise HTTPException(409, "a sweep is already running")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                fleet = (await client.get(rag + "/health")).json().get("student_fleet") or {}
        except Exception:
            fleet = {}
        copies = [c for c in fleet.get("copies", []) if c.get("alive")] if fleet.get("enabled") else []
        if not copies:
            raise HTTPException(400, "the sweep needs STUDENT_FLEET=on with live copies on this machine")
        slots = int(fleet.get("slots_per_copy") or 4)
        levels = sorted({1, max(1, slots // 2), slots})
        audit(request, "sweep", copies=len(copies), levels=levels)

        async def run() -> None:
            sweep.update(running=True, results=[], error=None, at=time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    for copy in copies:
                        for level in levels:
                            sweep["results"].append(await sweep_copy(client, f"http://127.0.0.1:{copy['port']}", level))
            except Exception as error:
                sweep["error"] = f"{type(error).__name__}: {error}"[:300]
            finally:
                sweep["running"] = False
        task = asyncio.get_running_loop().create_task(run())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"ok": True, "levels": levels, "copies": len(copies)}

    return app
