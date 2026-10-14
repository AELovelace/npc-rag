"""Exercise Windows installer decisions without installing packages or touching live services."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


POWERSHELL = shutil.which("powershell")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="Windows installer contract")
INSTALLER = Path(__file__).resolve().parents[1] / "ps/Install-Decider.ps1"


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.mark.parametrize("backend,mode,success", [
    ("Cuda", "ok", True), ("Vulkan", "ok", True),
    ("Cuda", "wheel_failure", False), ("Vulkan", "no_gpu", False),
])
def test_gpu_installer_replaces_cpu_build_and_requires_offload(tmp_path, backend, mode, success):
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "fake-python.ps1"
    fake.write_text("""
ConvertTo-Json -Compress -InputObject @($args) | Add-Content -LiteralPath $env:INSTALLER_TEST_LOG
$global:LASTEXITCODE = 0
if ($env:INSTALLER_TEST_MODE -eq 'wheel_failure' -and $args -contains '--force-reinstall') { $global:LASTEXITCODE = 1 }
if ($env:INSTALLER_TEST_MODE -eq 'no_gpu' -and ($args -join ' ') -match 'llama_supports_gpu_offload') { $global:LASTEXITCODE = 1 }
""", encoding="utf-8")
    # Mock directory creation: fake Python never uses the short build directory, so no C:\t writes occur.
    command = f"""
$ErrorActionPreference = 'Stop'
function New-Item {{ param($ItemType, $Path, [switch]$Force) }}
$originalTemp = $env:TEMP
$originalTmp = $env:TMP
$originalTmpDir = $env:TMPDIR
$failed = $false
try {{ & {_quote(INSTALLER)} -Python {_quote(fake)} -Backend {backend} -BuildTemp C:/t }}
catch {{ $failed = $true; Write-Output $_.Exception.Message }}
if ($env:TEMP -ne $originalTemp -or $env:TMP -ne $originalTmp -or $env:TMPDIR -ne $originalTmpDir) {{ exit 99 }}
if ($failed) {{ exit 1 }}
exit 0
"""
    env = {**os.environ, "INSTALLER_TEST_LOG": str(calls), "INSTALLER_TEST_MODE": mode}
    result = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if success else 1), result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]
    replacement = next(args for args in commands if "--force-reinstall" in args)
    assert "--only-binary=:all:" in replacement and "--no-deps" in replacement
    assert "--no-cache-dir" in replacement and "--extra-index-url" not in replacement
    assert replacement[replacement.index("--index-url") + 1].endswith("/cu124" if backend == "Cuda" else "/vulkan")
    assert "llama-cpp-python==0.3.36" in replacement
    assert not any("PackageNotFoundError" in " ".join(args) for args in commands)  # Same version cannot skip a GPU replacement.
    if mode == "wheel_failure":
        assert not any("-r" in args for args in commands)  # Do not conceal wheel failure with a CPU/source requirements install.
    else:
        assert any("-r" in args for args in commands)
        assert any("llama_backend_init()" in " ".join(args) for args in commands)
        assert commands[-1][-1] == backend
