"""Keep native warning/error diagnostics even when normal model-loading chatter is quiet."""
import os
import threading

_callback = None  # ctypes callbacks must stay alive as long as the native library can call them.
_state = threading.local()


def _forward(level, message, user_data):
    if level != 5:
        _state.important = level in (3, 4)  # ggml levels: WARN=3, ERROR=4, CONT=5.
    if message and getattr(_state, "important", False):
        try:
            os.write(2, message)  # Write directly before a native abort can discard buffered diagnostics.
        except OSError:
            pass  # A closed stderr must never raise through a C callback during shutdown.


def install_native_error_logging(llama):
    global _callback
    if _callback is None:
        _callback = llama.llama_log_callback(_forward)
    llama.llama_log_set(_callback, None)  # Reinstall if an earlier upstream Decider replaced the handler.
