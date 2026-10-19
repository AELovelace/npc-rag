"""Check or fast-forward this checkout to its configured upstream; preserve native GPU builds.

Run with the node's venv Python as the checkout owner. The control page starts the
dedicated systemd unit, so restarting either web service cannot interrupt an update.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATUS = "data/repo-update.json"
REQUIREMENTS = ("requirements.txt", "requirements-decider.txt")
PROTECTED = (".env", ".env.overflow", ".venv/", ".venv-train/", "data/models/", "data/index/", "data/gm-index/",
             "data/admin-audit.jsonl", STATUS, "data/repo-update.tmp", "data/repo-update.lock")


class UpdateError(RuntimeError):
    pass


def command(root: Path, args: list[str], *, timeout=120, label="Command") -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never", "PIP_NO_INPUT": "1"}
    try:
        result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UpdateError(f"{label} could not finish ({type(error).__name__}).") from None
    if result.returncode:
        raise UpdateError(f"{label} failed (exit {result.returncode}). Check repository access, dependencies and service permissions on this node.")
    return result.stdout.strip()  # Never publish raw stderr: remote URLs or package-index credentials may appear there.


def git(root: Path, *args: str) -> str:
    return command(root, ["git", *args], label="Git " + args[0])


def tracked_changes(root: Path) -> str:
    # Older checkouts tracked the append-only admin audit. Keep it locally, and separately reject any upstream change to it.
    return git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)data/admin-audit.jsonl")


@contextmanager
def exclusive(root: Path):
    path = root / "data/repo-update.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise UpdateError("Another repository check or update is already running.") from None
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)  # OS releases the lock even if the process crashes.


def inspect(root: Path, *, fetch=True) -> dict:
    branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    remote = git(root, "config", "--get", f"branch.{branch}.remote")
    merge = git(root, "config", "--get", f"branch.{branch}.merge")
    if not remote or remote == "." or remote.startswith("-") or not merge.startswith("refs/heads/"):
        raise UpdateError("Configure a remote tracking branch before updating this checkout.")
    if fetch:
        git(root, "fetch", "--no-tags", remote, merge)  # Read only the configured remote branch; the browser cannot choose a URL/ref.
    target = git(root, "rev-parse", "--verify", "FETCH_HEAD^{commit}" if fetch else "@{upstream}^{commit}")
    current = git(root, "rev-parse", "HEAD")
    dirty = bool(tracked_changes(root))
    ahead, behind = map(int, git(root, "rev-list", "--left-right", "--count", f"{current}...{target}").split())
    changes = git(root, "diff", "--name-only", current, target).splitlines()
    protected = any(p == name or (name.endswith("/") and p.startswith(name)) for p in changes for name in PROTECTED)
    system_changes = any(p.startswith("linux/") and p.endswith((".service", ".rules")) for p in changes)
    blocked = ("Local tracked changes must be committed or resolved first; no files were overwritten." if dirty else
               "Local commits are ahead of or diverge from upstream; reconcile them before updating." if ahead else
               "Upstream changes include node-local settings, models or indexes; review them manually." if protected else
               "System service definitions changed. Update manually and rerun linux/install-ubuntu.sh to install them." if system_changes else "")
    return {"branch": branch, "current": current, "target": target, "ahead": ahead, "behind": behind,
            "update_available": behind > 0, "blocked": blocked, "changed_files": changes}


def install_dependencies(root: Path, changed: list[str]) -> None:
    selected = [name for name in REQUIREMENTS if name in changed and (root / name).is_file()]
    if not selected:
        return  # Ordinary code fixes never reinstall or rebuild the ROCm/CUDA/Vulkan Python backend.
    try:
        native = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        native = None
    if "requirements-decider.txt" in selected and native is None:
        raise UpdateError("Decider dependencies changed but its native backend is not installed. Install the appropriate GPU backend manually.")
    with tempfile.TemporaryDirectory(prefix="npc-update-") as temporary:
        constraints = Path(temporary) / "native-constraints.txt"
        constraints.write_text(f"llama-cpp-python==={native}\n" if native else "", encoding="utf-8")
        args = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--constraint", str(constraints)]
        for name in selected:
            args.extend(["-r", name])
        command(root, [*args, "--dry-run"], timeout=300, label="Dependency resolution (native GPU version is pinned)")
        command(root, args, timeout=900, label="Dependency installation (native GPU version is pinned)")
    # No --upgrade/--force-reinstall: an already installed HIP build satisfying the pin is retained.


def restart_and_wait(root: Path, port: int, timeout=120) -> None:
    command(root, ["systemctl", "--no-ask-password", "restart", "npc-rag.service"], label="RAG restart")  # Wait for the old process to stop before checking readiness.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                health = json.load(response)
            if health.get("ok") and health.get("llm") and health.get("classifier"):
                fleet = health.get("student_fleet", {})
                if not fleet.get("enabled") or (fleet.get("copies") and all(c.get("alive") for c in fleet["copies"])):
                    return  # Readiness includes the classifier and every managed student copy, not just the HTTP listener.
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise UpdateError("Updated code installed, but RAG did not become healthy within 120 seconds. Inspect its journal; no automatic reset or rollback was performed.")


def update(root: Path, *, install=False, restart=False, port=9092) -> dict:
    state = {"running": True, "phase": "checking", "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}

    def record(**values):
        state.update(values)
        if not state["running"]:
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        target = root / STATUS
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, target)  # Polling clients never read half of a status document.
        print(state["phase"], state.get("message", ""), flush=True)

    with exclusive(root):
        record()
        try:
            info = inspect(root)
            record(**info)
            if info["blocked"]:
                raise UpdateError(info["blocked"])
            if not info["update_available"] or not install:
                record(running=False, phase="available" if info["update_available"] else "current",
                       message="Update available." if info["update_available"] else "Already up to date.")
                return state
            record(phase="installing", message="Installing the checked upstream revision.")
            # Recheck before touching files; never discard a concurrent edit or move a locally advanced branch.
            if git(root, "rev-parse", "HEAD") != info["current"] or tracked_changes(root):
                raise UpdateError("Checkout changed during the update check. Retry after resolving local work.")
            git(root, "-c", "core.hooksPath=" + str(root / "data/no-update-hooks"), "merge", "--ff-only", info["target"])
            record(installed=info["target"], phase="dependencies", message="Checking changed Python dependencies.")
            install_dependencies(root, info["changed_files"])
            command(root, [sys.executable, "-m", "compileall", "-q", "npc_rag", "python"], label="Python syntax check")
            command(root, [sys.executable, "-c", "from npc_rag.config import load_settings; load_settings()"], label="Node settings validation")
            if restart:
                record(phase="restarting", message="Reloading RAG and waiting for model readiness.")
                restart_and_wait(root, port)
                record(phase="restarting_admin", message="RAG is healthy; reloading the control page. Sign in again after it reconnects.")
                command(root, ["systemctl", "--no-ask-password", "--no-block", "restart", "npc-rag-admin.service"], label="Control page restart")
            record(running=False, phase="complete", message="Update installed; services restarted." if restart else "Update installed. Restart the services to use it.")
        except Exception as error:
            record(running=False, phase="failed", message=str(error) if isinstance(error, UpdateError) else f"Update failed ({type(error).__name__}).")
            raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--install", action="store_true")
    parser.add_argument("--restart-services", action="store_true")
    parser.add_argument("--port", type=int, default=9092)
    args = parser.parse_args()
    if args.restart_services and (not args.install or not sys.platform.startswith("linux")):
        parser.error("--restart-services requires --install on the Linux systemd node")
    try:
        update(ROOT, install=args.install, restart=args.restart_services, port=args.port)
    except Exception as error:
        print(str(error) if isinstance(error, UpdateError) else f"Update failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
