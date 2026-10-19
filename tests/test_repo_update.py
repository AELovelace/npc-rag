"""Updater integration with disposable Git remotes; never touches the real node or its services."""
import json
from pathlib import Path
import subprocess

import pytest

from python import update_repo as updater


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def commit(root, file, text):
    target = root / file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    git(root, "add", file)
    git(root, "commit", "-m", "fixture update")


@pytest.fixture
def repos(tmp_path):
    remote, author, node = (tmp_path / n for n in ("remote.git", "author", "node"))
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(tmp_path, "clone", str(remote), str(author))
    for key, value in (("user.name", "Updater Test"), ("user.email", "test@example.invalid")):
        git(author, "config", key, value)
    commit(author, ".gitignore", ".env\ndata/repo-update*\n__pycache__/\n*.pyc\n")
    commit(author, "npc_rag/__init__.py", "")
    commit(author, "npc_rag/config.py", "def load_settings(): return None\n")
    commit(author, "python/__init__.py", "")
    commit(author, "app.txt", "old\n")
    commit(author, "data/admin-audit.jsonl", "old audit\n")
    git(author, "push", "-u", "origin", "main")
    git(tmp_path, "clone", str(remote), str(node))
    for key, value in (("user.name", "Updater Test"), ("user.email", "test@example.invalid")):
        git(node, "config", key, value)
    (node / ".env").write_text("NODE_SECRET=preserve-me\n", encoding="utf-8")
    return author, node


def advance(author, file="app.txt", text="new\n"):
    commit(author, file, text)
    git(author, "push")


def test_check_then_install_then_noop_preserve_settings_and_audit(repos):
    author, node = repos
    advance(author)
    (node / "data/admin-audit.jsonl").write_text("local audit additions\n", encoding="utf-8")
    check = updater.update(node)
    assert check["phase"] == "available" and check["behind"] == 1
    assert (node / "app.txt").read_text() == "old\n"
    installed = updater.update(node, install=True)
    assert installed["phase"] == "complete"
    assert git(node, "rev-parse", "HEAD") == git(author, "rev-parse", "HEAD")
    assert (node / ".env").read_text() == "NODE_SECRET=preserve-me\n"
    assert (node / "data/admin-audit.jsonl").read_text() == "local audit additions\n"
    assert updater.update(node, install=True)["phase"] == "current"


@pytest.mark.parametrize("kind", ["dirty", "diverged", "collision", "protected", "units"])
def test_unsafe_updates_never_overwrite_local_files(repos, kind):
    author, node = repos
    before = git(node, "rev-parse", "HEAD")
    if kind == "dirty":
        (node / "app.txt").write_text("local changes\n")
        advance(author)
    elif kind == "diverged":
        commit(node, "app.txt", "local commit\n")
        before = git(node, "rev-parse", "HEAD")
        advance(author)
    elif kind == "collision":
        (node / "new.txt").write_text("my untracked file\n")
        advance(author, "new.txt", "remote file\n")
    elif kind == "protected":
        advance(author, "data/admin-audit.jsonl", "remote audit replacement\n")
    else:
        advance(author, "linux/npc-rag.service", "new service unit\n")
    with pytest.raises(updater.UpdateError):
        updater.update(node, install=True)
    assert git(node, "rev-parse", "HEAD") == before
    assert json.loads((node / updater.STATUS).read_text())["phase"] == "failed"
    assert (node / ".env").read_text() == "NODE_SECRET=preserve-me\n"
    if kind == "collision":
        assert (node / "new.txt").read_text() == "my untracked file\n"


def test_exclusive_lock_is_released_after_failure(tmp_path):
    with updater.exclusive(tmp_path):
        with pytest.raises(updater.UpdateError, match="already running"):
            with updater.exclusive(tmp_path):
                pass
    with updater.exclusive(tmp_path):
        pass


def test_dependency_installer_pins_existing_hip_build_without_force_reinstall(tmp_path, monkeypatch):
    (tmp_path / "requirements-decider.txt").write_text("llama-cpp-python==0.3.36\n")
    monkeypatch.setattr(updater.importlib.metadata, "version", lambda name: "0.3.36")
    calls = []
    def run(root, args, **kwargs):
        constraints = Path(args[args.index("--constraint") + 1]).read_text()
        calls.append((args, constraints))
        return ""
    monkeypatch.setattr(updater, "command", run)
    updater.install_dependencies(tmp_path, ["requirements-decider.txt"])
    assert len(calls) == 2 and "--dry-run" in calls[0][0]
    assert all(c == "llama-cpp-python===0.3.36\n" for _, c in calls)
    assert all("--upgrade" not in a and "--force-reinstall" not in a for a, _ in calls)


def test_failed_dependency_install_keeps_revision_evidence_without_restarting(repos, monkeypatch):
    author, node = repos
    advance(author, "requirements.txt", "nonexistent-test-package\n")
    def fail(*args):
        raise updater.UpdateError("Dependency installation failed")
    monkeypatch.setattr(updater, "install_dependencies", fail)
    monkeypatch.setattr(updater, "restart_and_wait", lambda *args: pytest.fail("must not restart after failed dependency install"))
    with pytest.raises(updater.UpdateError, match="Dependency"):
        updater.update(node, install=True, restart=True)
    result = json.loads((node / updater.STATUS).read_text())
    assert result["phase"] == "failed" and result["current"] != result["installed"]


def test_restart_waits_for_classifier_and_all_copies(monkeypatch, tmp_path):
    import io
    calls, delays = [], []
    reports = iter([
        {"ok": True, "llm": True, "classifier": False},
        {"ok": True, "llm": True, "classifier": True, "student_fleet": {"enabled": True, "copies": [{"alive": False}]}},
        {"ok": True, "llm": True, "classifier": True, "student_fleet": {"enabled": True, "copies": [{"alive": True}]}},
    ])
    monkeypatch.setattr(updater, "command", lambda root, args, **kwargs: calls.append(args))
    monkeypatch.setattr(updater.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(next(reports)).encode()))
    monkeypatch.setattr(updater.time, "sleep", lambda seconds: delays.append(seconds))
    updater.restart_and_wait(tmp_path, 9092)
    assert calls == [["systemctl", "--no-ask-password", "restart", "npc-rag.service"]]
    assert delays == [2, 2]


def test_restart_timeout_is_reported_without_rollback(monkeypatch, tmp_path):
    ticks = iter([0, 121])
    monkeypatch.setattr(updater.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(updater, "command", lambda *args, **kwargs: "")
    with pytest.raises(updater.UpdateError, match="no automatic reset or rollback"):
        updater.restart_and_wait(tmp_path, 9092)
