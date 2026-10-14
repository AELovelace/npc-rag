"""Install the pinned LiDollQuest Decider release before the native model becomes ready."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import os
import tempfile
import time
import zipfile

import httpx

from .config import ROOT
from .decider_prompt import checkpoint_layout

LOG = logging.getLogger(__name__)
CUSTOM_MODEL = "lidoll/decider0-1"
CUSTOM_GGUF = "decider0-1.gguf"
CONFIG_DIRECTORY = Path(__file__).with_name("decider_config")
RELEASE_MANIFEST = Path(__file__).with_name("decider_release.json")
CONFIG_FILES = ("decider_config.json", "routing_policy.json", "config.json",
                "tokenizer_config.json", "chat_template.jinja")


def _repository_configs() -> dict[str, bytes]:
    """Read tunable files from this Git checkout, never from the hosted support archive."""
    configs = {name: (CONFIG_DIRECTORY / name).read_bytes() for name in CONFIG_FILES}
    for name, content in configs.items():
        if name.endswith(".json"):
            json.loads(content)  # Reject broken edits before downloading or touching the working cache.
    decider = json.loads(configs["decider_config.json"])
    temperature = float(decider["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Decider temperature must be finite and positive")
    if checkpoint_layout(decider) != "state_first" or decider.get("layout") != "plain":
        raise ValueError("decider0-1 weights require the trained plain, state-first prompt layout")
    return configs


def _matches(path: Path, spec: dict) -> bool:
    """Check bytes as well as the digest; an interrupted download cannot pass as a model."""
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)  # Stream the 1.27 GB model without allocating it all in RAM.
    return digest.hexdigest() == spec["sha256"]


@contextmanager
def _install_lock(path: Path, timeout: float = 1200):
    """Serialize launches across processes; the OS releases the lock even after a crash."""
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()  # Windows byte locks require a byte to lock.
        deadline = time.monotonic() + timeout
        waiting = False
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for another Decider installation") from None
                if not waiting:
                    LOG.info("Waiting for another process to finish installing Decider")
                    waiting = True
                time.sleep(.25)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _download(client: httpx.Client, url: str, target: Path, spec: dict) -> None:
    """Download into staging, with one retry for transport failures and strict size/hash checks."""
    for attempt in range(2):
        LOG.info("Downloading %s (%.1f MB)", target.name, spec["bytes"] / 1_000_000)
        try:
            received, next_report = 0, 128 * 1024 * 1024
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with target.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        received += len(chunk)
                        if received > spec["bytes"]:
                            raise RuntimeError(f"Download exceeds the pinned size: {target.name}")
                        output.write(chunk)
                        if received >= next_report:
                            LOG.info("%s: %.0f / %.0f MB", target.name, received / 1_000_000,
                                     spec["bytes"] / 1_000_000)
                            next_report += 128 * 1024 * 1024
            break
        except httpx.TransportError:
            if attempt:
                raise
            LOG.warning("Download interrupted; retrying %s from the beginning", target.name)
    if not _matches(target, spec):
        raise RuntimeError(f"Size/SHA-256 verification failed: {target.name}")


def _unpack(archive: Path, stage: Path, specs: dict) -> None:
    """Extract only the expected flat files, validating each against the checked-in release."""
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) != len(specs) or {entry.filename for entry in entries} != set(specs):
            raise RuntimeError("Decider support archive has unexpected or duplicate files")
        for entry in entries:
            name = entry.filename
            if "/" in name or "\\" in name or ":" in name or name in (".", ".."):
                raise RuntimeError("Decider support archive contains an unsafe path")
            if entry.file_size != specs[name]["bytes"]:
                raise RuntimeError(f"Unexpected support file size: {name}")
            # Write plain files ourselves; ZIP permissions, links and paths are never followed.
            with bundle.open(entry) as source, (stage / name).open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            if not _matches(stage / name, specs[name]):
                raise RuntimeError(f"Support file SHA-256 verification failed: {name}")


def ensure_custom_decider(destination: Path | None = None, *, release_manifest=None, config_loader=None,
                          gguf_file=None, model_name='decider0-1') -> Path:
    """Reuse a verified local release offline, or download and repair missing/corrupt files."""
    release = json.loads((release_manifest or RELEASE_MANIFEST).read_text(encoding="utf-8"))
    gguf_file = gguf_file or CUSTOM_GGUF
    destination = (destination or ROOT / 'data/models' / model_name).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        configs = (config_loader or _repository_configs)()
        specs = dict(release["files"])
        for name, content in configs.items():
            specs[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        # Weights/assets have fixed release hashes; configuration follows the current Git checkout.
        with _install_lock(destination.parent / f".{destination.name}.lock"):
            LOG.info("Checking local Decider release at %s", destination)
            missing = {name for name, spec in specs.items()
                       if not _matches(destination / name, spec)}
            if not missing:
                LOG.info("Decider release verified; using local files (no download)")
                return destination
            # Only files created inside this unique temporary directory are cleaned up on failure.
            with tempfile.TemporaryDirectory(prefix=".download-", dir=destination) as temporary:
                stage = Path(temporary)
                with httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=True) as client:
                    support = {name: spec for name, spec in release["files"].items() if name != gguf_file}
                    if (missing - configs.keys()) & support.keys():
                        archive = stage / "support.zip"
                        _download(client, release["support_url"], archive, release["support_archive"])
                        _unpack(archive, stage, support)
                    if gguf_file in missing:
                        _download(client, release["download_url"], stage / gguf_file,
                                  release["files"][gguf_file])
                for name in missing & configs.keys():
                    (stage / name).write_bytes(configs[name])  # Repo config wins even when the ZIP carries older copies.
                # Publish only fully verified files. Existing valid files are never replaced.
                for name in sorted(missing):
                    (stage / name).replace(destination / name)
            LOG.info("Custom Decider installed and verified at %s", destination)
            return destination
    except (OSError, ValueError, KeyError, RuntimeError, httpx.HTTPError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Custom Decider startup check failed: {exc}. Ensure both {gguf_file} and "
            f"{model_name}-support.zip are hosted at https://lidoll.dev/gguf/, check "
            "the model's config directory in this checkout, and restart. "
            "No base-model fallback was loaded."
        ) from exc
