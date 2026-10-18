"""Student fleet: pinned GGUF download, device parsing, port planning, launching copies and restarting crashed ones."""
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from npc_rag import student_fleet as fleet_mod
from npc_rag.config import load_settings

FAKE_SERVER = '''
import argparse, http.server, sys, threading, time
p = argparse.ArgumentParser(); p.add_argument("--port", type=int); p.add_argument("--die-after", type=float, default=0)
args, _ = p.parse_known_args()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H)
if args.die_after:
    threading.Timer(args.die_after, lambda: (server.shutdown(), sys.exit(3))).start()
server.serve_forever()
sys.exit(3 if args.die_after else 0)
'''


def manifest(tmp_path, data: bytes, url="https://gguf.test/m.gguf"):
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"default": "m", "releases": {"m": {"file": "m.gguf", "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "download_url": url}}}), encoding="utf-8")
    return path


def test_release_defaults_and_rejects_unknown_names():
    spec = fleet_mod.release()
    assert spec["file"].endswith("-Q8_0.gguf") and spec["download_url"].startswith("https://lidoll.dev/gguf/")
    with pytest.raises(ValueError, match="Unknown STUDENT_MODEL"):
        fleet_mod.release("lidoll-fm-9.9")


def test_ensure_student_downloads_once_then_reuses_and_rejects_bad_bytes(tmp_path):
    data, calls = b"GGUF" + b"x" * 5000, []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=data)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = fleet_mod.ensure_student("", tmp_path / "models" / "m", manifest(tmp_path, data), client=client)
    assert path.read_bytes() == data and len(calls) == 1
    fleet_mod.ensure_student("", tmp_path / "models" / "m", manifest(tmp_path, data), client=client)
    assert len(calls) == 1, "a verified file is reused offline"
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"GGUF" + b"y" * 5000)))
    with pytest.raises(RuntimeError, match="verification"):
        fleet_mod.ensure_student("", tmp_path / "models" / "other", manifest(tmp_path, data), client=bad)
    assert not (tmp_path / "models" / "other" / "m.gguf").exists(), "nothing unverified is published"


def test_list_devices_reads_llama_server_output_and_skips_cpu():
    out = subprocess.CompletedProcess([], 0, "Available devices:\n  Vulkan0: AMD Radeon RX 6800 XT (16368 MiB, 15569 MiB free)\n"
                                             "  Vulkan1: AMD Radeon RX 6800 XT (16368 MiB, 15569 MiB free)\n  CPU0: Intel (65448 MiB)\n", "")
    devices = fleet_mod.list_devices("llama-server", runner=lambda *a, **k: out)
    assert [d["name"] for d in devices] == ["Vulkan0", "Vulkan1"] and devices[0]["mib"] == 16368


def test_plan_spreads_copies_and_skips_reserved_and_busy_ports():
    copies = fleet_mod.plan(["Vulkan0", "Vulkan1"], 2, 9090, reserved={9092}, free=lambda p: p != 9093)
    assert [(c.device, c.port) for c in copies] == [("Vulkan0", 9090), ("Vulkan0", 9091), ("Vulkan1", 9094), ("Vulkan1", 9095)]


def test_command_has_slots_context_device_and_extra_flags(settings):
    s = dataclasses.replace(settings, llama_server="llama-server.exe", student_slots=4, student_ctx_per_slot=8192,
                            student_args="--flash-attn on --cache-ram 0", host="0.0.0.0")
    cmd = fleet_mod.command(s, Path("m.gguf"), fleet_mod.Copy("Vulkan1", 9093))
    assert cmd[:3] == ["llama-server.exe", "-m", "m.gguf"]
    joined = " ".join(cmd)
    for part in ("--port 9093", "-c 32768", "-np 4", "--n-gpu-layers 999", "--device Vulkan1", "--flash-attn on", "--cache-ram 0"):
        assert part in joined
    assert "--cache-type" not in joined, "no KV-cache quantization by default (it cost speed on Vulkan)"


def test_fleet_launches_copies_restarts_a_crashed_one_and_stops_them(settings, tmp_path):
    script = tmp_path / "fake_llama.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    s = dataclasses.replace(settings, student_fleet=True, llama_server=str(script), student_model="", student_copies_per_device=1,
                            student_slots=2, student_base_port=19490, student_args="", port=19492)
    popen = lambda cmd, **kw: subprocess.Popen([sys.executable, *cmd], **kw)  # The fake "binary" is a Python script.
    fleet = fleet_mod.StudentFleet(s, model=tmp_path / "m.gguf", devices=["Vulkan0", "Vulkan1"], popen=popen,
                                   logs=tmp_path / "logs", pidfile=tmp_path / "fleet.json")
    try:
        fleet.start(timeout=60)
        assert fleet.urls == "http://127.0.0.1:19490,http://127.0.0.1:19491"
        assert fleet_mod.CURRENT is fleet and all(c["alive"] for c in fleet.view()["copies"])
        assert json.loads((tmp_path / "fleet.json").read_text())["ports"] == [19490, 19491]
        victim = fleet.copies[0]
        victim.process.kill()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (victim.restarts and fleet._ready(victim)):
            time.sleep(.5)
        assert victim.restarts == 1 and fleet._ready(victim), "the watchdog restarted the crashed copy"
    finally:
        fleet.stop()
    assert all(c.process.poll() is not None for c in fleet.copies) and fleet_mod.CURRENT is None
    assert not (tmp_path / "fleet.json").exists()


def test_clean_leftovers_only_stops_our_llama_server(tmp_path, monkeypatch):
    pidfile = tmp_path / "fleet.json"
    pidfile.write_text(json.dumps({"pids": [11, 22]}))
    monkeypatch.setattr(fleet_mod, "_image", lambda pid: {11: r"C:\llama\llama-server.exe", 22: r"C:\Windows\notepad.exe"}[pid])
    killed = []
    assert fleet_mod.clean_leftovers("C:/Scripts/llama/Vulkan/llama-server.exe", pidfile, kill=killed.append) == 1
    assert killed == [11] and not pidfile.exists()


def test_llama_server_is_found_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDENT_FLEET", "on")
    monkeypatch.setenv("LLAMA_SERVER", "")
    assert load_settings().student_fleet, "an empty LLAMA_SERVER is allowed: the fleet looks for one at start"
    monkeypatch.setattr("shutil.which", lambda name: None)
    (tmp_path / "CUDA").mkdir(); (tmp_path / "Vulkan").mkdir()
    for build in ("CUDA", "Vulkan"):
        (tmp_path / build / "llama-server.exe").write_bytes(b"")
    found = fleet_mod.find_llama_server("", patterns=(str(tmp_path / "*" / "llama-server.exe"),))
    assert Path(found).parent.name == "Vulkan", "a Vulkan build is preferred when several are installed"
    with pytest.raises(RuntimeError, match="no llama-server was found"):
        fleet_mod.find_llama_server("", patterns=(str(tmp_path / "missing" / "*.exe"),))
    with pytest.raises(RuntimeError, match="does not exist"):
        fleet_mod.find_llama_server(str(tmp_path / "nope.exe"))
