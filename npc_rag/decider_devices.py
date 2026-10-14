"""Device enumeration and model pinning for the pinned llama-cpp-python 0.3.36 ABI."""
from contextlib import contextmanager
import ctypes
from pathlib import Path


class _Caps(ctypes.Structure):
    _fields_ = [(name, ctypes.c_bool) for name in
                ("async_ops", "host_buffer", "buffer_from_host_ptr", "events", "mmap_support")]


class _Props(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("description", ctypes.c_char_p),
                ("memory_free", ctypes.c_size_t), ("memory_total", ctypes.c_size_t),
                ("type", ctypes.c_int), ("device_id", ctypes.c_char_p), ("caps", _Caps)]


class NativeDevices:
    """Read llama.cpp's own registry so CPU/BLAS adapters are never counted as cards."""

    def __init__(self):
        import llama_cpp
        from llama_cpp._ctypes_extensions import load_shared_library
        if llama_cpp.__version__ != "0.3.36":
            raise RuntimeError("Automatic Decider GPU assignment requires the pinned llama-cpp-python 0.3.36 wheel")
        self.llama = llama_cpp
        llama_cpp.llama_backend_init()  # Loads the installed wheel's CUDA/Vulkan backend before enumeration.
        folder = Path(llama_cpp.__file__).parent / "lib"
        self.libs = [load_shared_library(name, folder) for name in ("ggml", "ggml-base")]
        self.count = self._bind("ggml_backend_dev_count", [], ctypes.c_size_t)
        self.get = self._bind("ggml_backend_dev_get", [ctypes.c_size_t], ctypes.c_void_p)
        self.props = self._bind("ggml_backend_dev_get_props", [ctypes.c_void_p, ctypes.POINTER(_Props)], None)

    def _bind(self, name, args, result):
        for lib in self.libs:
            function = getattr(lib, name, None)
            if function is not None:
                function.argtypes, function.restype = args, result
                return function
        raise RuntimeError(f"Installed native library lacks {name}; install the pinned Decider wheel")

    def rows(self):
        rows = []
        for index in range(self.count()):
            pointer = self.get(index)
            props = _Props()
            self.props(pointer, ctypes.byref(props))
            if props.type not in (1, 2):
                continue  # ggml device types: 0 CPU, 1 dedicated GPU, 2 integrated GPU, 3 accelerator.
            rows.append({"name": props.name.decode(), "description": props.description.decode(),
                         "device_id": props.device_id.decode() if props.device_id else None,
                         "integrated": props.type == 2, "memory_total": props.memory_total,
                         "memory_free": props.memory_free, "pointer": pointer})
        return rows


def detect_devices():
    return [{k: v for k, v in row.items() if k != "pointer"} for row in NativeDevices().rows()]


def select_devices(rows, selection):
    """Choose one backend per physical card; explicit names can opt into an integrated GPU."""
    if selection == "auto":
        selected, seen = [], set()
        for row in rows:
            identity = row.get("device_id") or row["name"]
            if row.get("integrated") or identity in seen:
                continue
            selected.append(row)
            seen.add(identity)  # A wheel exposing CUDA and Vulkan for the same PCI card gets one worker.
    else:
        by_name = {row["name"]: row for row in rows}
        names = selection.split(",")
        missing = [name for name in names if name not in by_name]
        if missing:
            raise RuntimeError(f"Unknown Decider GPU(s): {', '.join(missing)}; available: {', '.join(by_name)}")
        selected = [by_name[name] for name in names]
        ids = [row.get("device_id") or row["name"] for row in selected]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Decider GPU selection contains the same physical card more than once")
    if not selected:
        raise RuntimeError("No dedicated Decider GPUs detected; install the GPU wheel, select a device explicitly, or set DECIDER_GPU_LAYERS=0")
    return selected


@contextmanager
def pinned_device(name):
    """Constrain the upstream engine's load to one device inside its isolated worker process."""
    registry = NativeDevices()
    row = next((r for r in registry.rows() if r["name"] == name), None)
    if row is None:
        raise RuntimeError(f"Decider GPU disappeared before model load: {name}")
    llama = registry.llama
    devices = (ctypes.c_void_p * 2)(row["pointer"], None)  # Native ABI requires a NULL-terminated device list.
    original = llama.llama_model_load_from_file

    def load(path, params):
        params.devices = ctypes.cast(devices, ctypes.c_void_p).value
        params.split_mode = llama.LLAMA_SPLIT_MODE_NONE
        params.main_gpu = 0  # Index in the restricted one-device list, not the host-wide GPU index.
        return original(path, params)

    llama.llama_model_load_from_file = load
    try:
        yield {k: v for k, v in row.items() if k != "pointer"}
    finally:
        llama.llama_model_load_from_file = original  # Never change inference or other models' load behavior.
