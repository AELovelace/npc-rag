"""Settings for every npc-rag service, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Repository root: data/, prompts and the .env file live here.


def _load_dotenv(path: Path) -> None:
    """Copy KEY=value lines from .env into os.environ without overriding variables that are already set."""
    if not path.is_file():
        return  # A .env file is optional; plain environment variables work on their own.
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue  # Skip blanks, comments and malformed lines.
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")  # Allow quoted values like NPC_NAME="Pip".
        os.environ.setdefault(key.strip(), value)  # Real environment variables win over the file.


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()  # Every setting is a trimmed string before conversion.


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default=False) -> bool:
    value = _env(name, 'true' if default else 'false').lower()
    if value not in ('true', 'false', '1', '0', 'on', 'off'):
        raise ValueError(f'{name} must be true or false')
    return value in ('true', '1', 'on')


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    if not 1 <= value <= 64:
        raise ValueError(f"{name} must be an integer from 1 to 64")
    return value  # Reject invalid capacity instead of silently creating an unlimited paid queue.


def _positive_float(name: str, default: float) -> float:
    value = _env_float(name, default)
    if not math.isfinite(value) or not 0 < value <= 120:
        raise ValueError(f"{name} must be greater than zero and at most 120 seconds")
    return value


def _path(name: str, default: str) -> Path:
    p = Path(_env(name, default))
    return p if p.is_absolute() else ROOT / p  # Relative paths are relative to the repo, not the working directory.


@dataclass(frozen=True)
class Settings:
    # ── Network layout ────────────────────────────────────────────────────
    llm_url: str                 # llama.cpp server that writes the NPC's reply (Sakura's 9090).
    classifier_url: str          # Small llama.cpp model that answers GAME or CHAT (Sakura's 9091).
    host: str                    # Bind address for npc-rag.
    port: int                    # 9092: /v1/npc/chat for the game server, plus /v1/search and /v1/reindex.
    api_key: str                 # Shared secret callers must send; empty disables the check.
    llm_api_key: str             # Optional bearer token for llama.cpp's --api-key (both model servers).

    # ── Wiki index ────────────────────────────────────────────────────────
    wiki_source: str             # Public wiki URL, or a local web/wiki folder with pages.json + content/.
    wiki_public_url: str         # Base used for deep links shown to players.
    index_dir: Path              # Where chunks.json / vectors.npy / meta.json are written.
    embed_model: str             # fastembed model name (ONNX, runs on CPU).
    embed_cache: Path            # Model download cache.
    embed_threads: int           # onnxruntime threads; 0 lets it decide.
    chunk_words: int             # Soft maximum words per chunk.
    min_relevance: float         # Dense score below which retrieval counts as "nothing relevant".

    # ── Classifier ────────────────────────────────────────────────────────
    classifier_model: str        # Model name sent to the classifier server.
    classifier_timeout: float    # Total classifier-model wait (slot queue + HTTP), then use the wiki score.
    rag_game_evidence: float     # Wiki score that immediately makes a message a game question.

    # ── Reply generation ──────────────────────────────────────────────────
    npc_name: str                # Default NPC name when the caller does not send one.
    persona_path: Path           # Editable persona text for the system prompt.
    llm_model: str               # Model name sent to llama.cpp (it serves whatever is loaded).
    max_tokens: int              # Reply length cap in tokens.
    temperature: float
    guard_keep_margin: float     # Sections scoring this far below the best one are not shown to the model.
    guard_min_words: int         # Shorter sections (chapter overviews) never count as notes.
    guard_retries: int           # Extra attempts when a reply names things no source mentions.
    guard_allowed_terms: tuple[str, ...]  # Names a reply may use even though no source mentions them.
    max_reply_chars: int       # Hard cap so the reply fits the in-game dialogue box.
    llm_timeout: float           # Seconds to wait for llama.cpp before using a fallback line.
    llm_concurrency: int         # Simultaneous llama.cpp requests (the server has 4 slots).
    top_k: int                   # Wiki excerpts given to the LLM for a game question.
    max_message_chars: int       # Longest player message accepted.
    session_turns: int           # Remembered messages per player (user + NPC lines).
    session_ttl: int             # Seconds of silence before a player's memory is forgotten.

    # ── Companions (/v1/companion/*) ──────────────────────────────────────
    companion_lore_k: int = 2             # Backstory sections shown per reply.
    companion_lore_min: float = 0.5       # Section relevance below which it stays out of the prompt.
    companion_memory_k: int = 4           # Shared memories shown per reply.
    companion_memory_min: float = 0.45    # Memory relevance floor (bge-small scores unrelated text around 0.4).
    companion_gossip_min: float = 0.45    # Anonymous gossip relevance floor.
    companion_max_reply_chars: int = 240  # Companion lines appear in area chat: keep them short.
    companion_embed_cache: int = 5000     # Memory texts whose vectors are kept between chats.
    companion_max_tokens: int = 100       # Generation cap for companions (replies are cut to 240 characters anyway).
    companion_note_k: int = 2             # Wiki notes a companion's game answer may use (Pip keeps RAG_TOP_K).
    companion_note_words: int = 120       # Each companion note is cut to this many words: prompt reading is most of a reply's time.
    ambient_local_reserve: int = 2        # Local slots ambient NPC lines must leave free for players (AMBIENT_NPC_LIFE_PLAN.md).
    openai_api_key: str = field(default="", repr=False)  # Keep the provider secret out of Settings diagnostics.
    openai_model: str = "gpt-6-luna"       # Explicit requested API model; configurable for later evaluation.
    openai_local_capacity: int = 4        # Total active local generations on this RAG process before cloud overflow.
    openai_concurrency: int = 4           # Maximum concurrent cloud generations per RAG process.
    openai_timeout: float = 12.0          # Cloud deadline, bounded further by the caller's reply timeout.
    classifier_cache_size: int = 1024    # Maximum completed labels and shared in-flight classifier calls; zero disables.
    classifier_cache_ttl: float = 300.0  # Seconds to reuse a model decision; wiki evidence is still checked afresh.
    classifier_backend: str = "llamacpp"  # Existing HTTP classifier or in-process Decider GGUF scoring.
    decider_model: str = "lidoll/decider0-1"  # Managed release alias, Hugging Face repo ID or local model directory.
    decider_gguf_file: str = "decider0-1.gguf"  # The managed alias always uses its pinned filename.
    decider_threads: int = 4             # CPU threads per native inference worker.
    decider_gpu_layers: int = 0          # CPU by default; -1 offloads all layers with a compatible build.
    decider_gpu_devices: str = "auto"    # GPU mode: one worker per detected card, or explicit CUDA0,Vulkan0 names.
    decider_context: int = 4096          # Entire prompt capacity, including the decision instructions.
    participation_enabled: bool = False  # Opt-in second model for JOIN/QUIET, served on this same HTTP port.
    participation_model: str = 'lidoll/decider-chat-0-1'
    participation_timeout: float = 10.0
    rag_keyword_coverage: float = 0.5    # Share of a message's meaningful words one wiki chunk must hold for the keyword rule; above 1 disables it.
    rag_keyword_dense: float = 0.62      # Wiki score the keyword rule still requires (below RAG_GAME_EVIDENCE, above unrelated chat).

    def __post_init__(self):
        if not self.participation_model.strip() or not math.isfinite(self.participation_timeout) or not 0 < self.participation_timeout <= 120:
            raise ValueError('PARTICIPATION_MODEL and a timeout from 0 to 120 seconds are required')
        if self.classifier_backend not in ("llamacpp", "decider"):
            raise ValueError("CLASSIFIER_BACKEND must be llamacpp or decider")
        if not self.decider_model.strip() or not self.decider_gguf_file.endswith(".gguf"):
            raise ValueError("DECIDER_MODEL and a DECIDER_GGUF_FILE ending in .gguf are required")
        if not 1 <= self.decider_threads <= 64 or not -1 <= self.decider_gpu_layers <= 256:
            raise ValueError("DECIDER_THREADS must be 1-64 and DECIDER_GPU_LAYERS must be -1 through 256")
        if not 2048 <= self.decider_context <= 32768:
            raise ValueError("DECIDER_CONTEXT must be 2048-32768 tokens")
        devices = self.decider_gpu_devices.split(",")
        if (not all(d.strip() and d == d.strip() for d in devices) or len(set(devices)) != len(devices)
                or ("auto" in devices and len(devices) != 1)):
            raise ValueError("DECIDER_GPU_DEVICES must be auto or a comma-separated list of unique device names")
        if not (math.isfinite(self.rag_keyword_coverage) and math.isfinite(self.rag_keyword_dense)
                and 0 < self.rag_keyword_coverage and 0 <= self.rag_keyword_dense <= 1):
            raise ValueError("RAG_KEYWORD_COVERAGE must be above 0 (above 1 disables it) and RAG_KEYWORD_DENSE must be 0-1")


def load_settings(env_file: str | None = None) -> Settings:
    """Build Settings from the environment, loading ROOT/.env first."""
    selected = Path(env_file) if env_file else _path("NPC_ENV_FILE", ".env")
    _load_dotenv(selected if selected.is_absolute() else ROOT / selected)  # A separate file lets both services share this checkout.
    return Settings(
        llm_url=_env("LLM_URL", "http://127.0.0.1:9090").rstrip("/"),  # A comma list spreads replies over several copies (llm.make_llm).
        classifier_url=_env("CLASSIFIER_URL", "http://127.0.0.1:9091").rstrip("/"),
        host=_env("NPC_HOST", "0.0.0.0"),
        port=_env_int("NPC_PORT", 9092),
        api_key=_env("NPC_API_KEY", ""),
        llm_api_key=_env("LLM_API_KEY", ""),
        wiki_source=_env("WIKI_SOURCE", "https://lidoll.dev/wiki/"),
        wiki_public_url=_env("WIKI_PUBLIC_URL", "https://lidoll.dev/wiki/"),
        index_dir=_path("INDEX_DIR", "data/index"),
        embed_model=_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        embed_cache=_path("EMBED_CACHE", "data/models"),
        embed_threads=_env_int("EMBED_THREADS", 0),
        chunk_words=_env_int("CHUNK_WORDS", 180),
        min_relevance=_env_float("MIN_RELEVANCE", 0.60),
        classifier_model=_env("CLASSIFIER_MODEL", "local"),
        classifier_timeout=_env_float("CLASSIFIER_TIMEOUT", 10.0),
        rag_game_evidence=_env_float("RAG_GAME_EVIDENCE", 0.66),
        npc_name=_env("NPC_NAME", "Pip"),
        persona_path=_path("PERSONA_FILE", "data/persona.md"),
        llm_model=_env("LLM_MODEL", "local"),
        max_tokens=_env_int("LLM_MAX_TOKENS", 220),
        temperature=_env_float("LLM_TEMPERATURE", 0.6),
        guard_keep_margin=_env_float("GUARD_KEEP_MARGIN", 0.06),
        guard_min_words=_env_int("GUARD_MIN_WORDS", 15),
        guard_retries=_env_int("GUARD_RETRIES", 1),
        guard_allowed_terms=tuple(t.strip() for t in _env("GUARD_ALLOWED_TERMS", "").split(",") if t.strip()),
        max_reply_chars=_env_int("MAX_REPLY_CHARS", 600),
        llm_timeout=_env_float("LLM_TIMEOUT", 45.0),
        llm_concurrency=_env_int("LLM_CONCURRENCY", 2),
        top_k=_env_int("RAG_TOP_K", 4),
        max_message_chars=_env_int("MAX_MESSAGE_CHARS", 500),
        session_turns=_env_int("SESSION_TURNS", 6),
        session_ttl=_env_int("SESSION_TTL", 1800),
        companion_lore_k=_env_int("COMPANION_LORE_K", 2),
        companion_lore_min=_env_float("COMPANION_LORE_MIN", 0.5),
        companion_memory_k=_env_int("COMPANION_MEMORY_K", 4),
        companion_memory_min=_env_float("COMPANION_MEMORY_MIN", 0.45),
        companion_gossip_min=_env_float("COMPANION_GOSSIP_MIN", 0.45),
        companion_max_reply_chars=_env_int("COMPANION_MAX_REPLY_CHARS", 240),
        companion_embed_cache=_env_int("COMPANION_EMBED_CACHE", 5000),
        companion_max_tokens=_env_int("COMPANION_MAX_TOKENS", 100),
        companion_note_k=_env_int("COMPANION_NOTE_K", 2),
        companion_note_words=_env_int("COMPANION_NOTE_WORDS", 120),
        ambient_local_reserve=_env_int("AMBIENT_LOCAL_RESERVE", 2),
        openai_api_key=_env("OPENAI_API_KEY", ""),
        openai_model=_env("OPENAI_MODEL", "gpt-6-luna"),
        openai_local_capacity=_positive_int("OPENAI_LOCAL_CAPACITY", 4),
        openai_concurrency=_positive_int("OPENAI_CONCURRENCY", 4),
        openai_timeout=_positive_float("OPENAI_TIMEOUT", 12.0),
        classifier_cache_size=_env_int("CLASSIFIER_CACHE_SIZE", 1024),
        classifier_cache_ttl=_env_float("CLASSIFIER_CACHE_TTL", 300.0),
        classifier_backend=_env("CLASSIFIER_BACKEND", "llamacpp").lower(),
        decider_model=_env("DECIDER_MODEL", "lidoll/decider0-1"),
        decider_gguf_file=_env("DECIDER_GGUF_FILE", "decider0-1.gguf"),
        decider_threads=_env_int("DECIDER_THREADS", 4),
        decider_gpu_layers=_env_int("DECIDER_GPU_LAYERS", 0),
        decider_gpu_devices=",".join(d.strip() for d in _env("DECIDER_GPU_DEVICES", "auto").split(",")),
        decider_context=_env_int("DECIDER_CONTEXT", 4096),
        participation_enabled=_env_bool('PARTICIPATION_ENABLED'),
        participation_model=_env('PARTICIPATION_MODEL', 'lidoll/decider-chat-0-1'),
        participation_timeout=_positive_float('PARTICIPATION_TIMEOUT', 10.0),
        rag_keyword_coverage=_env_float("RAG_KEYWORD_COVERAGE", 0.5),
        rag_keyword_dense=_env_float("RAG_KEYWORD_DENSE", 0.62),
    )
