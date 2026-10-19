"""GPU temps, VRAM and RAM for /health and the control page (npc_rag/hardware.py)."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

from npc_rag import hardware


def fake_amdgpu(root, index, used_gib, temps, busy=37, power_uw=180_500_000, pci="0x73bf"):
    """One amdgpu card the way the driver lays it out under /sys/class/drm."""
    dev = root / f"card{index}" / "device"
    hw = dev / "hwmon" / f"hwmon{index + 3}"
    hw.mkdir(parents=True)
    (dev / "mem_info_vram_total").write_text(str(16 * 2**30))
    (dev / "mem_info_vram_used").write_text(str(int(used_gib * 2**30)))
    (dev / "gpu_busy_percent").write_text(str(busy))
    (dev / "device").write_text(pci)
    for n, (label, milli) in enumerate(temps.items(), start=1):
        (hw / f"temp{n}_label").write_text(label)
        (hw / f"temp{n}_input").write_text(str(milli))
    (hw / "power1_average").write_text(str(power_uw))


def test_amd_cards_read_temps_vram_busy_and_power_from_sysfs(tmp_path):
    fake_amdgpu(tmp_path, 0, 12.5, {"edge": 61000, "junction": 78000, "mem": 70000})
    fake_amdgpu(tmp_path, 1, 4.0, {"edge": 55000})
    (tmp_path / "card0-DP-1").mkdir()  # A display connector, not a GPU.
    (tmp_path / "card2" / "device").mkdir(parents=True)  # Not amdgpu: no VRAM files.
    cards = hardware.amd_cards(tmp_path)
    assert [c["index"] for c in cards] == [0, 1]
    a = cards[0]
    assert a["name"] == "Radeon RX 6800 / 6800 XT / 6900 XT" and a["vendor"] == "amd"
    assert (a["temp_c"], a["hotspot_c"], a["mem_temp_c"]) == (61.0, 78.0, 70.0)
    assert (a["vram_used_mb"], a["vram_total_mb"], a["vram_pct"]) == (12800, 16384, 78.1)
    assert a["busy_pct"] == 37 and a["power_w"] == 180.5
    assert cards[1]["hotspot_c"] is None, "a missing sensor is None, not an error"


def test_nvidia_cards_parse_one_query_and_tolerate_na(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    out = "NVIDIA GeForce GTX TITAN X, 71, 9000, 12288, 88, [N/A]\nNVIDIA GeForce GTX TITAN X, 64, 1200, 12288, 5, 98.25\n"
    cards = hardware.nvidia_cards(runner=lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    assert [(c["temp_c"], c["vram_used_mb"], c["busy_pct"], c["power_w"]) for c in cards] == [(71.0, 9000, 88, None), (64.0, 1200, 5, 98.2)]
    assert hardware.nvidia_cards(runner=lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "boom")) == []


def test_ram_reports_used_and_free():
    mem = hardware.ram()
    assert mem is None or (mem["total_gb"] > 0 and 0 <= mem["used_pct"] <= 100 and mem["used_gb"] <= mem["total_gb"])


def test_snapshot_is_cached_and_a_hung_reading_never_breaks_health(monkeypatch):
    calls = []
    monkeypatch.setattr(hardware, "_cache", None)
    monkeypatch.setattr(hardware, "collect", lambda: calls.append(1) or {"gpus": [], "ram": None})
    hardware.snapshot()
    hardware.snapshot()
    assert len(calls) == 1, "one reading serves every /health call for CACHE_S seconds"
    monkeypatch.setattr(hardware, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(hardware, "_cached_at", 0.0)  # Due for a fresh reading.
    monkeypatch.setattr(hardware, "collect", lambda: time.sleep(1) or {})
    got = asyncio.run(hardware.snapshot_async())
    assert "error" in got and got["gpus"] == [], "the last reading comes back with a note instead of an exception"
