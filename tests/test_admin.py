"""The npc-rag control page (ADMIN_PORT 9091): login, write-only secrets, validated .env edits, restart and status."""
from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from npc_rag import admin

PASSWORD = "correct horse battery"
POST = {"X-NPC-Admin": "1"}  # The page's own header; a cross-site form can't send it.
ENV = """# Model servers
LLM_URL=http://127.0.0.1:9090
# Shared secret
NPC_API_KEY=super-secret-value-123
CLASSIFIER_BACKEND=llamacpp
"""


@pytest.fixture
def page(settings, tmp_path):
    env = tmp_path / ".env"
    env.write_text(ENV, encoding="utf-8")
    s = dataclasses.replace(settings, admin_password=PASSWORD, port=1, admin_restart_cmd="restart-npc-rag --now")
    app = admin.create_admin_app(s, str(env), audit_file=tmp_path / "audit.jsonl")
    return TestClient(app), env, tmp_path / "audit.jsonl"


def login(client, password=PASSWORD):
    return client.post("/api/login", json={"password": password}, headers=POST)


def test_the_page_refuses_to_start_without_an_admin_password(settings):
    with pytest.raises(ValueError, match="ADMIN_PASSWORD"):
        admin.create_admin_app(dataclasses.replace(settings, admin_password=""))
    with pytest.raises(ValueError, match="8\+"):
        admin.create_admin_app(dataclasses.replace(settings, admin_password="short"))


def test_everything_needs_a_login_and_wrong_passwords_are_rate_limited(page):
    client, _, audit = page
    assert client.get("/").status_code == 200, "the page itself loads so it can show the login form"
    assert client.get("/api/status").status_code == 401 and client.get("/api/settings").status_code == 401
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 403, "login needs the page header too"
    for _ in range(admin.FAILS):
        assert login(client, "nope").status_code == 401
    assert login(client).status_code == 429, "five misses lock the address out for a while, even with the right password"
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["action"] for r in rows] == ["login_failed"] * admin.FAILS and "nope" not in audit.read_text()


def test_a_login_cookie_is_signed_httponly_and_strict(page):
    client, _, _ = page
    r = login(client)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    client.cookies.set(admin.COOKIE, "9999999999.forged")
    assert client.get("/api/settings").status_code == 401, "a cookie this process didn't sign is refused"


def test_secrets_are_write_only_and_edits_keep_comments_with_a_backup(page):
    client, env, audit = page
    login(client)
    got = client.get("/api/settings")
    assert "super-secret-value-123" not in got.text
    rows = {f["key"]: f for f in got.json()["fields"]}
    assert rows["NPC_API_KEY"] == {**rows["NPC_API_KEY"], "set": True} and "value" not in rows["NPC_API_KEY"]
    assert rows["LLM_URL"]["value"] == "http://127.0.0.1:9090"
    assert client.post("/api/settings", json={"changes": {"LLM_URL": "x"}}).status_code == 403, "no page header, no write"
    r = client.post("/api/settings", json={"changes": {"LLM_URL": "http://127.0.0.1:9100", "STUDENT_FLEET": "on"}}, headers=POST)
    assert r.status_code == 200 and r.json()["restart_needed"]
    text = env.read_text(encoding="utf-8")
    assert "LLM_URL=http://127.0.0.1:9100" in text and "# Model servers" in text and "NPC_API_KEY=super-secret-value-123" in text
    assert text.rstrip().endswith("STUDENT_FLEET=on"), "a key the file lacked is appended"
    assert (env.parent / r.json()["backup"]).read_text(encoding="utf-8") == ENV
    assert "9100" not in audit.read_text(), "the audit log names keys, never values"


def test_bad_edits_are_refused_and_the_file_is_untouched(page):
    client, env, _ = page
    login(client)
    for changes, why in (({"ADMIN_PASSWORD": "x" * 9}, "can't be edited"), ({"NPC_HOST": "1.2.3.4"}, "can't be edited"),
                         ({"CLASSIFIER_BACKEND": "gpt"}, "one of"), ({"STUDENT_SLOTS": "four"}, "whole number"),
                         ({"LLM_URL": "a\nNPC_API_KEY=stolen"}, "one line"),
                         ({"DECIDER_THREADS": "99"}, "DECIDER_THREADS must be 1-64")):  # The last one only load_settings catches.
        r = client.post("/api/settings", json={"changes": changes}, headers=POST)
        assert r.status_code == 400 and why in r.json()["detail"], (changes, r.text)
    assert env.read_text(encoding="utf-8") == ENV and not list(env.parent.glob(".env.bak-*"))


def test_restart_runs_the_configured_command_and_status_survives_npc_rag_being_down(page, monkeypatch):
    client, _, _ = page
    ran = []
    monkeypatch.setattr(admin, "_run", lambda cmd, timeout=10: (ran.append(cmd), (0, ""))[1])
    login(client)
    assert client.post("/api/restart", headers=POST).json() == {"ok": True}
    assert ran[-1][0] == "restart-npc-rag" and ran[-1][-1] == "--now"
    status = client.get("/api/status").json()
    assert "not answering" in status["health"]["error"] and status["restart_available"]


def test_render_env_replaces_in_place_and_ignores_commented_examples():
    text = "# STUDENT_FLEET=on (example)\nSTUDENT_FLEET=off\nA=1\n"
    assert admin.render_env(text, {"STUDENT_FLEET": "on"}) == "# STUDENT_FLEET=on (example)\nSTUDENT_FLEET=on\nA=1\n"


def test_linux_restart_is_systemctl_no_block(settings, monkeypatch):
    monkeypatch.setattr(admin, "LINUX", True)
    cmd = admin.restart_command(dataclasses.replace(settings, admin_restart_cmd="", admin_service="npc-rag.service"))
    assert cmd == ["systemctl", "--no-ask-password", "--no-block", "restart", "npc-rag.service"]
