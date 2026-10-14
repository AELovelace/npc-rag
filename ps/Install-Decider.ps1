<#
.SYNOPSIS
    Install Decider in the existing RAG environment using a short Windows build path.
.PARAMETER Backend
    Auto preserves an installed native build. Cuda or Vulkan replaces it with the matching official GPU wheel.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Install-Decider.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Install-Decider.ps1 -Backend Cuda
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ps\Install-Decider.ps1 -Backend Vulkan
#>
param(
    [string]$Python = "",     # Defaults to the existing npc-rag virtual environment.
    [string]$BuildTemp = "",  # Defaults to a one-letter folder at the checkout drive root, such as C:\t.
    [ValidateSet('Auto', 'Cuda', 'Vulkan')]
    [string]$Backend = 'Auto'
)
$ErrorActionPreference = "Stop"

function Initialize-DeciderCompiler {
    # A normal PowerShell session does not inherit Visual Studio's compiler/library environment.
    if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and (Get-Command nmake.exe -ErrorAction SilentlyContinue)) { return }
    $vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vsWhere -PathType Leaf) {
        $vsInstall = & $vsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $vsInstall) {
            $devShell = Join-Path ($vsInstall | Select-Object -First 1) 'Common7\Tools\Launch-VsDevShell.ps1'
            if (Test-Path -LiteralPath $devShell -PathType Leaf) {
                & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation # Load MSVC, NMake, headers and libraries together.
            }
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue) -or -not (Get-Command nmake.exe -ErrorAction SilentlyContinue)) {
        throw @'
MSVC C++ compiler/NMake are unavailable. Install Visual Studio 2022 Build Tools with
"Desktop development with C++" (MSVC x64/x86 tools and a Windows SDK), then retry.
Direct installer (also works without winget on Windows LTSC):
https://aka.ms/vs/17/release/vs_BuildTools.exe
Run it and select "Desktop development with C++", keeping the recommended components.
Optional winget command when available:
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
If already installed, open "Developer PowerShell for VS 2022" and retry there.
'@
    }
}

$RagRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $RagRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "RAG Python was not found. Run ps\Install-NpcRag.ps1 first." }
if (-not $BuildTemp) { $BuildTemp = Join-Path ([IO.Path]::GetPathRoot($RagRoot)) 't' }
$BuildTemp = [IO.Path]::GetFullPath($BuildTemp)
# The pinned 0.3.36 archive has a 178-character member; pip adds 72 characters beneath TEMP.
# Keep the resulting filename below 260 characters, including its terminating null character.
if ($BuildTemp.Length -gt 8) {
    throw "Build path is too long for llama-cpp-python 0.3.36. Pass -BuildTemp with a writable absolute path of at most 8 characters, for example C:\t."
}
New-Item -ItemType Directory -Path $BuildTemp -Force | Out-Null # Keep extraction paths below Windows' legacy limit.
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
$oldTmpDir = $env:TMPDIR
try {
    $env:TEMP = $BuildTemp
    $env:TMP = $BuildTemp
    $env:TMPDIR = $BuildTemp # Python chooses its temp directory when the new pip process starts.
    & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Decider needs Python 3.11+ in this virtual environment." }
    $nativeCheck = @'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    sys.exit(0 if version('llama-cpp-python') == '0.3.36' else 1)
except PackageNotFoundError:
    sys.exit(1)
'@
    if ($Backend -eq 'Auto') {
        & $Python -c $nativeCheck
        if ($LASTEXITCODE -ne 0) { Initialize-DeciderCompiler } # The original source-install path still checks its toolchain.
    } else {
        $wheelBackend = if ($Backend -eq 'Cuda') { 'cu124' } else { 'vulkan' }
        $wheelIndex = "https://abetlen.github.io/llama-cpp-python/whl/$wheelBackend"
        Write-Host "Replacing llama-cpp-python with the official $Backend 0.3.36 wheel. Stop this RAG service first."
        # Use only this backend's index, and force replacement even when the CPU build has the same version.
        # No dependency reinstall here: the requirements step below supplies any missing Python packages.
        & $Python -m pip install --force-reinstall --no-cache-dir --no-deps --only-binary=:all: `
            --index-url $wheelIndex 'llama-cpp-python==0.3.36'
        if ($LASTEXITCODE -ne 0) {
            throw "$Backend wheel installation failed. Stop RAG to release native DLLs; verify Windows x64/Python compatibility and access to $wheelIndex. No CPU/source fallback was attempted."
        }
    }
    Write-Host "Installing Decider with temporary build directory $BuildTemp"
    & $Python -m pip install --no-cache-dir -r (Join-Path $RagRoot 'requirements-decider.txt')
    if ($LASTEXITCODE -ne 0) { throw "Decider installation failed; see the pip error above. Extraction errors are separate from compiler errors; source compilation needs Visual Studio C++ Build Tools and CMake." }
    & $Python -c "from decider.infer import Decider; import llama_cpp" # Verify native imports without loading or downloading weights.
    if ($LASTEXITCODE -ne 0) { throw "Decider native imports failed; resolve the dependency/DLL error before enabling the backend." }
    $gpuCheck = @'
import sys
import llama_cpp
llama_cpp.llama_backend_init()  # Initialize registered backends before asking about available GPU offload.
try:
    supported = bool(llama_cpp.llama_supports_gpu_offload())
    print('GPU offload supported:', supported)
    print(llama_cpp.llama_print_system_info().decode('utf-8', errors='replace'))
    sys.exit(0 if sys.argv[1] == 'Auto' or supported else 1)
finally:
    llama_cpp.llama_backend_free()
'@
    & $Python -c $gpuCheck $Backend
    if ($LASTEXITCODE -ne 0) {
        throw "$Backend was requested but GPU offload is unavailable. Check the GPU driver and native DLL dependencies before setting DECIDER_GPU_LAYERS=-1."
    }
    if ($Backend -ne 'Auto') {
        Write-Host "GPU backend verified. Set DECIDER_GPU_LAYERS=-1 in .env, restart RAG, and verify model offloading with GPU monitoring."
    }
    Write-Host "Decider dependencies installed. Run the classifier evaluation before enabling the backend." -ForegroundColor Green
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    $env:TMPDIR = $oldTmpDir # Restore the caller's environment on success, failure or interruption.
}
