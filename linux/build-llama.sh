#!/usr/bin/env bash
# Builds llama.cpp's llama-server for an AMD node on Ubuntu 26.04 LTS, twice:
#   /opt/llama.cpp/hip/bin/llama-server     ROCm HIP (preferred by npc-rag on Linux)
#   /opt/llama.cpp/vulkan/bin/llama-server  RADV Vulkan (fallback / A-B comparison)
# npc-rag finds both on its own (LLAMA_SERVER empty); set LLAMA_SERVER in .env to pin one.
# Usage: sudo ./build-llama.sh            (LLAMA_TAG=b9481 GPU_TARGETS=gfx1030 by default)
#        sudo LLAMA_TAG=master ./build-llama.sh
#        sudo BUILDS=vulkan ./build-llama.sh
#        sudo ROCM_SOURCE=amd ./build-llama.sh   (ROCm already installed from AMD's repo into /opt/rocm: newer than the archive)
set -euo pipefail

PREFIX="${PREFIX:-/opt/llama.cpp}"      # Install root.
LLAMA_TAG="${LLAMA_TAG:-b9481}"         # Node A's tested build; any llama.cpp tag or branch works.
GPU_TARGETS="${GPU_TARGETS:-gfx1030}"   # RX 6800 / 6800 XT / 6900 XT. rocminfo | grep gfx shows yours.
BUILDS="${BUILDS:-hip vulkan}"          # Which builds to make.
ROCM_SOURCE="${ROCM_SOURCE:-archive}"   # archive = Ubuntu's own `apt install rocm` (ROCm 7.1 at 26.04 launch); amd = leave /opt/rocm alone.
SRC="$PREFIX/src"

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }

# Toolchain + Mesa RADV Vulkan (both in Ubuntu's main/universe archive).
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git cmake ninja-build build-essential libcurl4-openssl-dev libssl-dev \
  libvulkan-dev glslc spirv-headers mesa-vulkan-drivers vulkan-tools
# ROCm for the HIP build. Ubuntu 26.04 carries it in the archive; the -dev packages are what llama.cpp compiles against.
if [[ " $BUILDS " == *" hip "* && "$ROCM_SOURCE" == archive ]]; then
  apt-get install -y rocm || echo "WARNING: apt install rocm failed; the HIP build will be skipped."
  apt-get install -y hipcc libhipblas-dev librocblas-dev rocminfo rocm-smi || \
    echo "WARNING: some ROCm -dev packages are missing from the archive; try ROCM_SOURCE=amd after AMD's installer."
fi
export PATH="$PATH:/opt/rocm/bin"   # AMD's repo installs hipconfig here; the archive puts it in /usr/bin.

if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" fetch --tags origin
else
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" checkout --force "$LLAMA_TAG"
if [[ "$LLAMA_TAG" == master ]]; then
  git -C "$SRC" pull --ff-only   # A branch: bring it up to date.
fi

build() {  # build <name> <cmake flags...>: a static llama-server in $PREFIX/<name>/bin.
  local name="$1"; shift
  cmake -S "$SRC" -B "$SRC/build-$name" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "$@"
  cmake --build "$SRC/build-$name" --target llama-server -j "$(nproc)"
  install -D -m 755 "$SRC/build-$name/bin/llama-server" "$PREFIX/$name/bin/llama-server"
  echo "== $name: $("$PREFIX/$name/bin/llama-server" --version 2>&1 | head -1)"
  "$PREFIX/$name/bin/llama-server" --list-devices 2>&1 | sed 's/^/   /'
}

for b in $BUILDS; do
  case "$b" in
    hip)    # A HIP problem never blocks the Vulkan build: npc-rag runs fine on Vulkan alone.
            if ! command -v hipconfig >/dev/null; then
              echo "WARNING: no hipconfig (ROCm not installed): skipping the HIP build."; continue
            fi
            ( HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" build hip -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGETS" ) || \
              echo "WARNING: the HIP build failed (see above); Vulkan still builds. Newer ROCm: AMD's installer, then ROCM_SOURCE=amd." ;;
    vulkan) build vulkan -DGGML_VULKAN=ON ;;
    *)      echo "unknown build '$b' (hip or vulkan)"; exit 1 ;;
  esac
done
echo "Done. npc-rag prefers $PREFIX/hip when both exist; STUDENT_DEVICES=auto picks up ROCm0/ROCm1 (or Vulkan0/Vulkan1)."
