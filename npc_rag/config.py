"""Settings for every npc-rag service, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
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


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(_env(name, default))
    return p if p.is_absolute() else ROOT / p  # Relative paths are relative to the repo, not the working directory.


@dataclass(frozen=True)
class Settings:
    # ── Network layout ────────────────────────────────────────────────────
    llm_url: str                 # llama.cpp server that writes the NPC's reply (port 9090).
    rag_url: str                 # Where the agent reaches the RAG service (port 9092).
    agent_host: str              # Bind address for the classifier/agent service.
    agent_port: int              # 9091: classifier + the player-facing /v1/npc/chat endpoint.
    rag_host: str                # Bind address for the RAG service.
    rag_port: int                # 9092: wiki retrieval.
    api_key: str                 # Shared secret callers must send; empty disables the check.
    llm_api_key: str             # Optional bearer token for llama.cpp's --api-key.

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
    seeds_path: Path             # Labelled example messages the classifier compares against.
    classifier_margin: float     # kNN margin needed to decide without asking for more evidence.
    rag_game_evidence: float     # Wiki score that makes an unsure message count as a game question.
    llm_tiebreak: bool           # Ask the LLM for a one-word verdict when the CPU signals disagree.

    # ── Reply generation ──────────────────────────────────────────────────
    npc_name: str                # Default NPC name when the caller does not send one.
    persona_path: Path           # Editable persona text for the system prompt.
    llm_model: str               # Model name sent to llama.cpp (it serves whatever is loaded).
    max_tokens: int              # Reply length cap in tokens.
    temperature: float
    max_reply_chars: int         # Hard cap so the reply fits the in-game dialogue box.
    llm_timeout: float           # Seconds to wait for llama.cpp before using a fallback line.
    llm_concurrency: int         # Simultaneous llama.cpp requests (the server has 4 slots).
    top_k: int                   # Wiki excerpts given to the LLM for a game question.
    max_message_chars: int       # Longest player message accepted.
    session_turns: int           # Remembered messages per player (user + NPC lines).
    session_ttl: int             # Seconds of silence before a player's memory is forgotten.


def load_settings() -> Settings:
    """Build Settings from the environment, loading ROOT/.env first."""
    _load_dotenv(ROOT / ".env")
    rag_port = _env_int("RAG_PORT", 9092)
    return Settings(
        llm_url=_env("LLM_URL", "http://47.51.162.110:9090").rstrip("/"),
        rag_url=_env("RAG_URL", f"http://127.0.0.1:{rag_port}").rstrip("/"),
        agent_host=_env("AGENT_HOST", "0.0.0.0"),
        agent_port=_env_int("AGENT_PORT", 9091),
        rag_host=_env("RAG_HOST", "127.0.0.1"),
        rag_port=rag_port,
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
        seeds_path=_path("CLASSIFIER_SEEDS", "data/classifier_seeds.json"),
        classifier_margin=_env_float("CLASSIFIER_MARGIN", 0.04),
        rag_game_evidence=_env_float("RAG_GAME_EVIDENCE", 0.66),
        llm_tiebreak=_env_bool("CLASSIFIER_LLM_TIEBREAK", True),
        npc_name=_env("NPC_NAME", "Pip"),
        persona_path=_path("PERSONA_FILE", "data/persona.md"),
        llm_model=_env("LLM_MODEL", "local"),
        max_tokens=_env_int("LLM_MAX_TOKENS", 220),
        temperature=_env_float("LLM_TEMPERATURE", 0.6),
        max_reply_chars=_env_int("MAX_REPLY_CHARS", 600),
        llm_timeout=_env_float("LLM_TIMEOUT", 45.0),
        llm_concurrency=_env_int("LLM_CONCURRENCY", 2),
        top_k=_env_int("RAG_TOP_K", 4),
        max_message_chars=_env_int("MAX_MESSAGE_CHARS", 500),
        session_turns=_env_int("SESSION_TURNS", 6),
        session_ttl=_env_int("SESSION_TTL", 1800),
    )
