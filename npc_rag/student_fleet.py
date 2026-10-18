"""Student fleet (STUDENT_FLEET=on): npc-rag fetches the pinned lidoll-fm GGUF and runs its own llama-server copies.

Why several copies per card: llama-server feeds its GPU from one CPU thread. On node A's RX 6800 XTs (Vulkan) that thread
was pinned at 100% while the GPU waited (2026-10-18), capping one copy near 90 generated tokens/s however many slots it
had. Each extra process brings its own feeding thread, so two copies per card roughly double a node's throughput.

What it does at startup (cli serve, before the app is built):
  1. ensure_student(): download the release named in student_release.json from lidoll.dev/gguf once, verify bytes + sha256,
     then reuse it offline.
  2. Find the GPUs (`llama-server --list-devices`, or STUDENT_DEVICES), plan STUDENT_COPIES_PER_DEVICE copies per card on
     free ports from STUDENT_BASE_PORT (npc-rag's own port is never used), launch them, and wait until each /health is 200.
  3. Point npc-rag at them: LLM_URL becomes every copy, LLM_CONCURRENCY becomes STUDENT_SLOTS (each copy's -np).
  4. A watchdog restarts any copy that exits (backoff up to 60 s). On Windows the copies live in a job object, so they
     die with npc-rag even when it is force-killed; a pid file cleans up leftovers on the next start everywhere else.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import ROOT
from .model_download import _download, _install_lock, _matches

LOG = logging.getLogger(__name__)
RELEASES = Path(__file__).with_name("student_release.json")
PIDFILE = ROOT / "data" / "student-fleet.json"
CURRENT: "StudentFleet | None" = None  # The running fleet, shown on /health.
DEVICE_LINE = re.compile(r"^\s*([A-Za-z]+\d+)\s*:\s*(.+?)\s*\((\d+)\s*MiB")  # "  Vulkan0: AMD Radeon RX 6800 XT (16368 MiB, ...)"


def release(name: str = "", manifest: Path = RELEASES) -> dict:
    """The pinned release by name ("" = the manifest's default)."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    key = name or data["default"]
    if key not in data["releases"]:
        raise ValueError(f"Unknown STUDENT_MODEL {key!r}; student_release.json has: {', '.join(data['releases'])}")
    return {"name": key, **data["releases"][key]}


def ensure_student(name: str = "", destination: Path | None = None, manifest: Path = RELEASES, client=None) -> Path:
    """Reuse a verified local GGUF, else download it from lidoll.dev/gguf and verify bytes + SHA-256 before publishing it."""
    spec = release(name, manifest)
    folder = Path(destination or ROOT / "data" / "models" / spec["name"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    target, pin = folder / spec["file"], {"bytes": spec["bytes"], "sha256": spec["sha256"]}
    with _install_lock(folder.parent / f".{folder.name}.lock"):
        if _matches(target, pin):
            LOG.info("Student %s verified at %s (no download)", spec["name"], target)
            return target
        with tempfile.TemporaryDirectory(prefix=".download-", dir=folder) as temporary:
            stage = Path(temporary) / spec["file"]
            own = client is None
            client = client or httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=True)
            try:
                _download(client, spec["download_url"], stage, pin)  # Size and hash are checked before anything is published.
            finally:
                if own:
                    client.close()
            stage.replace(target)
    LOG.info("Student %s downloaded and verified at %s", spec["name"], target)
    return target


SEARCH = ("C:/Scripts/llama/*/llama-server.exe", "C:/Scripts/llama/llama-server.exe", "C:/Scripts/Sakura_LLAMA-Server/**/llama-server.exe",
          "C:/Scripts/llama.cpp/**/llama-server.exe", "/usr/local/bin/llama-server", "/opt/llama.cpp/**/llama-server")


def find_llama_server(configured: str = "", patterns=SEARCH) -> str:
    """LLAMA_SERVER if set, else llama-server on PATH, else the usual install folders (a Vulkan build is preferred on Windows)."""
    if configured:
        if not Path(configured).is_file():
            raise RuntimeError(f"LLAMA_SERVER={configured} does not exist")
        return configured
    import glob
    import shutil
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return found
    hits = sorted({h for p in patterns for h in glob.glob(p, recursive=True)}, key=lambda h: ("vulkan" not in h.lower(), h))
    if hits:
        return hits[0]
    raise RuntimeError("STUDENT_FLEET=on but no llama-server was found. Set LLAMA_SERVER=<path to llama-server.exe> in .env "
                       "(looked on PATH and in: " + ", ".join(patterns) + ")")


def list_devices(server: str, runner=subprocess.run) -> list[dict]:
    """GPUs as llama-server itself names them (Vulkan0, CUDA0, ...); CPU entries are skipped."""
    out = runner([server, "--list-devices"], capture_output=True, text=True, timeout=120)
    devices = []
    for line in f"{out.stdout}\n{out.stderr}".splitlines():
        m = DEVICE_LINE.match(line)
        if m and not m.group(1).upper().startswith("CPU"):
            devices.append({"name": m.group(1), "description": m.group(2), "mib": int(m.group(3))})
    return devices


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


@dataclass
class Copy:
    device: str
    port: int
    process: subprocess.Popen | None = None
    restarts: int = 0
    exited_at: float | None = None
    last_exit: int | None = None
    log: Path | None = None


def plan(devices: list[str], copies_per_device: int, base_port: int, reserved=(), free=port_free) -> list[Copy]:
    """copies_per_device copies on each device, on the first free ports from base_port (reserved ports never used)."""
    taken, port, out = set(reserved), base_port, []
    for device in devices:
        for _ in range(copies_per_device):
            while port in taken or not free(port):
                port += 1
            out.append(Copy(device, port))
            taken.add(port)
            port += 1
    return out


def command(settings, model: Path, copy: Copy) -> list[str]:
    """One llama-server copy: every layer on its card, slots x context, the fleet's extra flags."""
    return [settings.llama_server, "-m", str(model), "--host", settings.host, "--port", str(copy.port),
            "-c", str(settings.student_slots * settings.student_ctx_per_slot), "-np", str(settings.student_slots),
            "--n-gpu-layers", "999", "--device", copy.device, *shlex.split(settings.student_args, posix=os.name != "nt")]


class _Job:
    """Windows job object with KILL_ON_JOB_CLOSE: children die with npc-rag, even after Stop-Process -Force."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.CreateJobObjectW.restype = wintypes.HANDLE
        handle = self.k32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if handle and self.k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):  # 9 = ExtendedLimitInformation
            self.handle = handle
        else:
            LOG.warning("Could not create a job object; leftover llama-server copies are cleaned up via %s instead", PIDFILE)

    def add(self, process: subprocess.Popen) -> None:
        if self.handle:
            import ctypes
            from ctypes import wintypes
            self.k32.OpenProcess.restype = wintypes.HANDLE
            proc = self.k32.OpenProcess(0x0101, False, process.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
            if proc:
                self.k32.AssignProcessToJobObject(wintypes.HANDLE(self.handle), wintypes.HANDLE(proc))
                self.k32.CloseHandle(wintypes.HANDLE(proc))


def _image(pid: int) -> str:
    """The executable path of a process ('' if it is gone), so only our own llama-server leftovers are ever stopped."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            size, buf = wintypes.DWORD(1024), ctypes.create_unicode_buffer(1024)
            return buf.value if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) else ""
        finally:
            k32.CloseHandle(handle)
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def clean_leftovers(server: str, pidfile: Path = PIDFILE, kill=None) -> int:
    """Stop copies a previous npc-rag left running (only processes whose executable is our llama-server)."""
    if not pidfile.is_file():
        return 0
    stopped = 0
    try:
        pids = json.loads(pidfile.read_text(encoding="utf-8")).get("pids", [])
    except ValueError:
        pids = []
    for pid in pids:
        image = _image(int(pid))
        if image and Path(image).name.lower() == Path(server).name.lower():
            try:
                (kill or (lambda p: os.kill(p, 9)))(int(pid))
                stopped += 1
            except OSError:
                pass
    pidfile.unlink(missing_ok=True)
    if stopped:
        LOG.warning("Stopped %d llama-server copies left by a previous npc-rag run", stopped)
    return stopped


class StudentFleet:
    def __init__(self, settings, *, model: Path | None = None, devices: list[str] | None = None, popen=subprocess.Popen,
                 logs: Path | None = None, pidfile: Path = PIDFILE, ready_url=None):
        self.settings, self.model, self.devices, self.popen = settings, model, devices, popen
        self.logs, self.pidfile = Path(logs or ROOT / "logs"), Path(pidfile)
        self.ready_url = ready_url or (lambda c: f"http://127.0.0.1:{c.port}/health")
        self.copies: list[Copy] = []
        self.spec: dict = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._job = _Job()
        self._watch: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------------------------------------------
    def start(self, timeout: float = 600) -> "StudentFleet":
        global CURRENT
        from dataclasses import replace
        self.settings = replace(self.settings, llama_server=find_llama_server(self.settings.llama_server))
        s = self.settings
        LOG.info("Student fleet using %s", s.llama_server)
        self.spec = release(s.student_model)
        self.model = self.model or ensure_student(s.student_model)
        clean_leftovers(s.llama_server, self.pidfile)
        devices = self.devices or self._detect()
        self.copies = plan(devices, s.student_copies_per_device, s.student_base_port, reserved={s.port})
        self.logs.mkdir(parents=True, exist_ok=True)
        for copy in self.copies:
            self._spawn(copy)
        self._save_pids()
        try:
            self._wait_ready(timeout)
        except Exception:
            self.stop()
            raise
        self._watch = threading.Thread(target=self._watchdog, name="student-fleet-watchdog", daemon=True)
        self._watch.start()
        CURRENT = self
        LOG.info("Student fleet ready: %s on %s (%d copies x %d slots)", self.spec["name"], self.urls,
                 len(self.copies), s.student_slots)
        return self

    def stop(self) -> None:
        global CURRENT
        self._stop.set()
        with self._lock:
            for copy in self.copies:
                if copy.process and copy.process.poll() is None:
                    copy.process.terminate()
            for copy in self.copies:
                if copy.process:
                    try:
                        copy.process.wait(10)
                    except subprocess.TimeoutExpired:
                        copy.process.kill()
        self.pidfile.unlink(missing_ok=True)
        if CURRENT is self:
            CURRENT = None

    # -- what npc-rag uses ----------------------------------------------------------------------------------------------
    @property
    def urls(self) -> str:
        return ",".join(f"http://127.0.0.1:{c.port}" for c in self.copies)

    def view(self) -> dict:
        return {"model": self.spec.get("name"), "file": self.spec.get("file"), "slots_per_copy": self.settings.student_slots,
                "ctx_per_slot": self.settings.student_ctx_per_slot,
                "copies": [{"device": c.device, "port": c.port, "alive": bool(c.process and c.process.poll() is None),
                            "restarts": c.restarts, "last_exit": c.last_exit} for c in self.copies]}

    # -- internals ------------------------------------------------------------------------------------------------------
    def _detect(self) -> list[str]:
        if self.settings.student_devices != "auto":
            return [d for d in self.settings.student_devices.split(",") if d]
        found = [d["name"] for d in list_devices(self.settings.llama_server)]
        if not found:
            raise RuntimeError(f"{self.settings.llama_server} --list-devices shows no GPU; set STUDENT_DEVICES (e.g. Vulkan0)")
        return found

    def _spawn(self, copy: Copy) -> None:
        copy.log = self.logs / f"student-{copy.port}.log"
        with copy.log.open("ab") as log:  # The child keeps its own handle; ours closes right away.
            log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} start {self.spec.get('name')} on {copy.device}:{copy.port}\n".encode())
            log.flush()
            flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
            copy.process = self.popen(command(self.settings, self.model, copy), stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, creationflags=flags)
        self._job.add(copy.process)

    def _save_pids(self) -> None:
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(json.dumps({"pids": [c.process.pid for c in self.copies if c.process],
                                            "ports": [c.port for c in self.copies]}), encoding="utf-8")

    def _ready(self, copy: Copy) -> bool:
        try:
            return httpx.get(self.ready_url(copy), timeout=3).status_code == 200  # llama-server says 503 while loading.
        except httpx.HTTPError:
            return False

    def _wait_ready(self, timeout: float) -> None:
        deadline, waiting = time.monotonic() + timeout, list(self.copies)
        while waiting:
            for copy in list(waiting):
                if copy.process.poll() is not None:
                    raise RuntimeError(f"llama-server on {copy.device}:{copy.port} exited with {copy.process.returncode} "
                                       f"while loading; see {copy.log}")
                if self._ready(copy):
                    waiting.remove(copy)
            if waiting and time.monotonic() > deadline:
                raise RuntimeError(f"llama-server copies not ready after {timeout:.0f} s: "
                                   + ", ".join(f"{c.device}:{c.port}" for c in waiting))
            if waiting:
                time.sleep(1)

    def _watchdog(self, interval: float = 2.0) -> None:
        while not self._stop.wait(interval):
            with self._lock:
                for copy in self.copies:
                    code = copy.process.poll() if copy.process else 0
                    if code is None:
                        continue
                    now = time.monotonic()
                    if copy.exited_at is None:
                        copy.exited_at, copy.last_exit = now, code
                        LOG.warning("llama-server %s:%s exited with %s; restarting", copy.device, copy.port, code)
                    if now - copy.exited_at >= min(60.0, 5.0 * 2 ** min(copy.restarts, 4)):
                        copy.restarts += 1
                        copy.exited_at = None
                        self._spawn(copy)
                        self._save_pids()
