"""Gateway settings from environment variables or an optional .env file."""

from __future__ import annotations

from ipaddress import ip_address

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # HTTP listener and permitted browser origins.
    bind_host: str = Field(default="127.0.0.1", alias="BIND_HOST")
    bind_port: int = Field(default=8080, alias="BIND_PORT")
    allowed_origins: list[str] = Field(default_factory=list, alias="ALLOWED_ORIGINS")

    # Mock mode exercises the pipeline without model services.
    adapter_mode: str = Field(default="real", alias="ADAPTER_MODE")

    # OpenAI-compatible generation endpoint and conversation budget.
    llm_base_url: str = Field(default="http://127.0.0.1:8000", alias="LLM_BASE_URL")
    llm_model: str = Field(default="qwen3_30b_a3b", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_thinking_mode: str = Field(default="none", alias="LLM_THINKING_MODE")
    llm_max_tokens: int = Field(default=384, alias="LLM_MAX_TOKENS", ge=1)
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_history_max_tokens: int = Field(default=3000, alias="LLM_HISTORY_MAX_TOKENS", ge=1)
    prompt_context_file: str = Field(default="", alias="PROMPT_CONTEXT_FILE")

    # Audio rate the recognizer and the voice-activity detector work at.
    stt_sample_rate: int = Field(default=16000, alias="STT_SAMPLE_RATE")

    # Cached acknowledgement while the answer generates.
    ack_inject: bool = Field(default=True, alias="ACK_INJECT")
    ack_phrases: str = Field(default="", alias="ACK_PHRASES")
    ack_phrases_busy: str = Field(default="", alias="ACK_PHRASES_BUSY")
    ack_long_when_busy: int = Field(default=2, alias="ACK_LONG_WHEN_BUSY")
    ack_at_ms: int = Field(default=800, alias="ACK_AT_MS")

    # Streaming mono PCM synthesis.
    tts_base_url: str = Field(default="http://127.0.0.1:8082", alias="TTS_BASE_URL")
    tts_voice: str = Field(default="default", alias="TTS_VOICE")
    tts_sample_rate: int = Field(default=24000, alias="TTS_SAMPLE_RATE")

    # ICE discovery and relay credentials.
    stun_urls: str = Field(default="stun:stun.cloudflare.com:3478", alias="STUN_URLS")
    turn_url: str = Field(default="", alias="TURN_URL")
    turn_username: str = Field(default="", alias="TURN_USERNAME")
    turn_password: str = Field(default="", alias="TURN_PASSWORD")
    cf_turn_key_id: str = Field(default="", alias="CF_TURN_KEY_ID")
    cf_turn_api_token: str = Field(default="", alias="CF_TURN_API_TOKEN")
    cf_turn_ttl_s: int = Field(default=14400, alias="CF_TURN_TTL_S")

    # Session admission and concurrent generation limits.
    max_sessions: int = Field(default=16, alias="MAX_SESSIONS", ge=1)
    max_active_generations: int = Field(default=4, alias="MAX_ACTIVE_GENERATIONS", ge=1)
    session_creates_per_min: int = Field(default=0, alias="SESSION_CREATES_PER_MIN", ge=0)
    max_session_seconds: int = Field(default=4 * 3600, alias="MAX_SESSION_SECONDS", ge=1)
    max_utterance_seconds: float = Field(default=30.0, alias="MAX_UTTERANCE_SECONDS")
    max_text_input_chars: int = Field(default=2000, alias="MAX_TEXT_INPUT_CHARS", ge=1)

    # Opening buffer and recovery buffer after playback underruns.
    playback_first_prebuffer_ms: int = Field(default=120, alias="PLAYBACK_FIRST_PREBUFFER_MS")
    playback_rebuffer_ms: int = Field(default=240, alias="PLAYBACK_REBUFFER_MS")

    # Speech detection, endpointing and interruption thresholds.
    vad_aggressiveness: int = Field(default=2, alias="VAD_AGGRESSIVENESS")
    vad_pre_roll_ms: int = Field(default=300, alias="VAD_PRE_ROLL_MS")
    vad_start_ms: int = Field(default=120, alias="VAD_START_MS")
    vad_end_silence_ms: int = Field(default=550, alias="VAD_END_SILENCE_MS")
    speculative_silence_ms: int = Field(default=300, alias="SPECULATIVE_SILENCE_MS")
    # Streaming recognition (Moonshine v2): one single-thread instance per admitted session.
    stt_moonshine_v2_arch: str = Field(default="medium-streaming", alias="STT_MOONSHINE_V2_ARCH")
    stt_moonshine_v2_dir: str = Field(default="models/moonshine-v2", alias="STT_MOONSHINE_V2_DIR")
    semantic_end_silence_ms: int = Field(default=350, alias="SEMANTIC_END_SILENCE_MS")
    barge_in_confirm_ms: int = Field(default=250, alias="BARGE_IN_CONFIRM_MS")
    barge_in_min_rms: int = Field(default=350, alias="BARGE_IN_MIN_RMS")
    utterance_min_rms: int = Field(default=250, alias="UTTERANCE_MIN_RMS")

    # Suppress recognition of recent assistant audio picked up by the mic.
    echo_suppress_similarity: float = Field(default=0.72, alias="ECHO_SUPPRESS_SIMILARITY")
    echo_suppress_window_s: float = Field(default=15.0, alias="ECHO_SUPPRESS_WINDOW_S")

    # First-fragment and later-fragment speech boundaries.
    segment_min_words: int = Field(default=8, alias="SEGMENT_MIN_WORDS")
    segment_min_chars: int = Field(default=40, alias="SEGMENT_MIN_CHARS")
    segment_first_min_words: int = Field(default=4, alias="SEGMENT_FIRST_MIN_WORDS")
    segment_first_min_chars: int = Field(default=16, alias="SEGMENT_FIRST_MIN_CHARS")
    segment_max_chars: int = Field(default=140, alias="SEGMENT_MAX_CHARS")
    segment_max_wait_ms: int = Field(default=1500, alias="SEGMENT_MAX_WAIT_MS")

    # Static assets and access control.
    frontend_dist: str = Field(default="", alias="FRONTEND_DIST")

    auth_token: str = Field(default="", alias="AUTH_TOKEN")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def require_auth_on_remote_bind(self) -> "Settings":
        host = self.bind_host.strip()
        if host.lower() == "localhost":
            return self
        try:
            loopback = ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback and not self.auth_token:
            raise ValueError("AUTH_TOKEN is required when BIND_HOST is not loopback")
        return self


def ice_servers(settings: Settings) -> list[dict]:
    """ICE server list in the shape both browsers and aiortc understand."""
    servers: list[dict] = []
    stun = [u.strip() for u in settings.stun_urls.split(",") if u.strip()]
    if stun:
        servers.append({"urls": stun})
    if settings.turn_url:
        servers.append(
            {
                "urls": [settings.turn_url],
                "username": settings.turn_username,
                "credential": settings.turn_password,
            }
        )
    return servers


def load_settings() -> Settings:
    return Settings()


# Log only non-secret capacity and latency settings.
_EFFECTIVE_KEYS = (
    "adapter_mode",
    "llm_model",
    "llm_max_tokens",
    "llm_history_max_tokens",
    "llm_temperature",
    "prompt_context_file",
    "stt_moonshine_v2_arch",
    "max_sessions",
    "max_active_generations",
    "session_creates_per_min",
    "ack_inject",
    "ack_at_ms",
    "ack_long_when_busy",
    "vad_end_silence_ms",
    "speculative_silence_ms",
    "semantic_end_silence_ms",
    "segment_min_words",
    "segment_min_chars",
    "segment_first_min_words",
    "segment_first_min_chars",
    "segment_max_wait_ms",
    "playback_first_prebuffer_ms",
    "playback_rebuffer_ms",
)


def describe(settings: Settings) -> str:
    """One line naming the effective capacity-relevant configuration.

    Formatted as `key=value` pairs so a load test can lift it straight
    out of gateway.log and record it alongside the run's numbers.
    """
    parts = []
    for key in _EFFECTIVE_KEYS:
        value = getattr(settings, key)
        if isinstance(value, bool):
            value = "true" if value else "false"
        parts.append(f"{key}={value}")
    return " ".join(parts)
