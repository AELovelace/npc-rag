"""GPU temperatures, VRAM, busy % and system RAM for /health (9092) and the control page (9091).

Readings are cheap and cached for CACHE_S seconds, and collected in a worker thread with a time limit, so the game
server's /health probes never wait on a slow tool:
  * AMD on Linux: the amdgpu driver's own sysfs files (no rocm-smi or subprocess needed);
  * NVIDIA (Linux or Windows): one nvidia-smi query;
  * RAM: /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows.
Every card has the same keys whatever the vendor; a value the hardware doesn't report is None.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CACHE_S = 5.0          # One reading serves every /health call for this long.
TIMEOUT_S = 3.0        # /health never waits longer than this for a fresh reading (it returns the last one instead).
DRM = Path("/sys/class/drm")
AMD_NAMES = {"0x73bf": "Radeon RX 6800 / 6800 XT / 6900 XT", "0x73df": "Radeon RX 6700 XT", "0x744c": "Radeon RX 7900 XT / XTX",
             "0x7480": "Radeon RX 7600"}  # PCI device id -> name, for the cards we run; anything else shows its id.
_lock = threading.Lock()
_cache: dict | None = None
_cached_at = 0.0


def _num(path: Path) -> float | None:
    """A sysfs number, or None when the file is missing or unreadable."""
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def amd_cards(root: Path = DRM) -> list[dict]:
    """Every amdgpu card from sysfs: temperatures (edge, hotspot, memory), VRAM, busy % and power."""
    cards = []
    for card in sorted(root.glob("card[0-9]*"), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else 999):
        if not card.name[4:].isdigit():
            continue  # card0-DP-1 and friends are display connectors, not GPUs.
        dev = card / "device"
        total = _num(dev / "mem_info_vram_total")
        if total is None:
            continue  # Not an amdgpu device (or no VRAM info).
        used, temps, power = _num(dev / "mem_info_vram_used"), {}, None
        for hwmon in sorted((dev / "hwmon").glob("hwmon*")):
            for t in hwmon.glob("temp*_input"):
                label_file = t.with_name(t.name.replace("_input", "_label"))
                label = label_file.read_text().strip().lower() if label_file.is_file() else t.name
                value = _num(t)
                if value is not None:
                    temps[label] = round(value / 1000, 1)  # Millidegrees.
            watts = _num(hwmon / "power1_average")
            if watts is None:
                watts = _num(hwmon / "power1_input")
            if watts is not None:
                power = round(watts / 1e6, 1)  # Microwatts.
        pci = (dev / "device").read_text().strip() if (dev / "device").is_file() else ""
        product = (dev / "product_name").read_text().strip() if (dev / "product_name").is_file() else ""
        cards.append(_card(len(cards), "amd", product or AMD_NAMES.get(pci, f"AMD GPU {pci}".strip()),
                           temps.get("edge", next(iter(temps.values()), None)), temps.get("junction"), temps.get("mem"),
                           used, total, _num(dev / "gpu_busy_percent"), power))
    return cards


def nvidia_cards(runner=subprocess.run) -> list[dict]:
    """Every NVIDIA card from one nvidia-smi query; [] when nvidia-smi isn't installed or fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        done = runner([exe, "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode:
        return []
    def value(text):  # "[N/A]" and blanks become None.
        try:
            return float(text)
        except ValueError:
            return None
    cards = []
    for row in done.stdout.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 6:
            name, temp, used, total, util, power = parts
            mib = 2**20
            cards.append(_card(len(cards), "nvidia", name, value(temp), None, None,
                               None if value(used) is None else value(used) * mib, None if value(total) is None else value(total) * mib,
                               value(util), value(power)))
    return cards


def _card(index, vendor, name, temp, hotspot, mem_temp, used_b, total_b, busy, power) -> dict:
    """The shape every card has in /health."""
    used_mb = None if used_b is None else round(used_b / 2**20)
    total_mb = None if total_b is None else round(total_b / 2**20)
    return {"index": index, "vendor": vendor, "name": name, "temp_c": temp, "hotspot_c": hotspot, "mem_temp_c": mem_temp,
            "vram_used_mb": used_mb, "vram_total_mb": total_mb,
            "vram_pct": round(100 * used_mb / total_mb, 1) if used_mb is not None and total_mb else None,
            "busy_pct": None if busy is None else round(busy), "power_w": None if power is None else round(power, 1)}


def ram() -> dict | None:
    """System RAM in GB (total, available, used, used %), plus swap on Linux; None where unsupported."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        rows = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            if rest.strip():
                rows[key] = int(rest.split()[0]) * 1024  # kB -> bytes.
        total, avail = rows.get("MemTotal", 0), rows.get("MemAvailable", rows.get("MemFree", 0))
        swap = rows.get("SwapTotal", 0) - rows.get("SwapFree", 0)
        return _ram(total, avail, swap_used=swap)
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                        ("virt_total", ctypes.c_ulonglong), ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return _ram(status.total, status.avail)
    return None


def _ram(total, avail, swap_used=None) -> dict:
    gb = lambda b: round(b / 2**30, 1)  # noqa: E731
    return {"total_gb": gb(total), "available_gb": gb(avail), "used_gb": gb(total - avail),
            "used_pct": round(100 * (total - avail) / total, 1) if total else None,
            **({"swap_used_gb": gb(swap_used)} if swap_used is not None else {})}


def process_rss_mb() -> float | None:
    """This npc-rag process's resident memory (Linux)."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    return None


def collect() -> dict:
    """A fresh reading (blocking; runs in a worker thread)."""
    gpus = amd_cards() if sys.platform.startswith("linux") else []
    source = "sysfs" if gpus else ""
    if not gpus:
        gpus = nvidia_cards()
        source = "nvidia-smi" if gpus else "none"
    return {"at": round(time.time()), "gpus": gpus, "gpu_source": source, "ram": ram(), "npc_rag_rss_mb": process_rss_mb()}


def snapshot() -> dict:
    """The cached reading, refreshed when older than CACHE_S (blocking)."""
    global _cache, _cached_at
    with _lock:
        if _cache is None or time.monotonic() - _cached_at >= CACHE_S:
            _cache, _cached_at = collect(), time.monotonic()
        return _cache


async def snapshot_async() -> dict:
    """For /health: a fresh-enough reading within TIMEOUT_S, else the last one (or an error note), never an exception."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(snapshot), TIMEOUT_S)
    except Exception as error:  # A hung tool or odd sysfs must never break /health.
        return {**(_cache or {}), "error": f"hardware reading failed: {type(error).__name__}"}
