"""DECIDER_FLASH_ATTN for Decider engines, which build their own llama.cpp context from llama_context_default_params().

decider.engine_gguf.GGUFEngine takes no flash-attention flag, so the mode goes in through the defaults it reads.  "off" works
around llama-cpp-python 0.3.36 HIP builds aborting in the flash-attention tile kernel on RDNA2 (gfx1030):
GGML_ASSERT(max_blocks_per_sm > 0) in fattn-common.cuh at the first decode.
"""

MODES = {"auto": -1, "off": 0, "on": 1}  # llama_flash_attn_type: AUTO, DISABLED, ENABLED.
_original = None


def set_flash_attention(llama, mode: str) -> None:
    """Make llama_context_default_params() hand out this flash-attention mode ("auto" restores llama.cpp's own default)."""
    global _original
    base = getattr(llama, "llama_cpp", llama)  # The package re-exports llama_cpp.llama_cpp; patch both names.
    if _original is None:
        _original = base.llama_context_default_params
    value = MODES[mode]
    original = _original

    def defaults():
        params = original()
        fields = {name for name, _ in type(params)._fields_}
        if "flash_attn_type" in fields:
            params.flash_attn_type = value
        elif "flash_attn" in fields:  # Older llama.cpp: a plain on/off flag.
            params.flash_attn = value == MODES["on"]
        return params

    for module in (llama, base):
        module.llama_context_default_params = original if value == MODES["auto"] else defaults
