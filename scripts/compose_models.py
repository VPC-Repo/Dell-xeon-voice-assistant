#!/usr/bin/env python3
"""Fetch public model files and initialize private TURN credentials for Compose."""

from __future__ import annotations

import os
import re
import secrets
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

MODELS = Path("/models")
RUNTIME = Path("/runtime")
QWEN = "Qwen3-30B-A3B-Instruct-2507-FP8"
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


def download(url: str, target: Path, minimum_bytes: int) -> None:
    if target.is_file() and target.stat().st_size >= minimum_bytes:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    print(f"Downloading {target.name}...", flush=True)
    try:
        urlretrieve(url, temporary)
        if temporary.stat().st_size < minimum_bytes:
            raise RuntimeError(f"download too small: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def private_write(target: Path, content: str) -> None:
    """Replace a credential file atomically; it is private from creation."""
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def turn_config() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    secret_file = RUNTIME / "turn-password"
    if secret_file.exists():
        password = secret_file.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", password):
            raise RuntimeError("Invalid stored TURN password; restore the runtime volume")
        os.chmod(secret_file, 0o600)
    else:
        password = secrets.token_hex(32)
        private_write(secret_file, password + "\n")
    private_write(
        RUNTIME / "turn.env",
        "TURN_URL='turn:127.0.0.1:3478?transport=tcp'\n"
        "TURN_USERNAME=voice\n"
        f"TURN_PASSWORD={password}\n",
    )
    private_write(
        RUNTIME / "turnserver.conf",
        "listening-ip=127.0.0.1\nrelay-ip=127.0.0.1\n"
        "listening-port=3478\nmin-port=49160\nmax-port=49200\n"
        f"lt-cred-mech\nuser=voice:{password}\nrealm=voice-ssh\n"
        "fingerprint\nallow-loopback-peers\nno-tls\nno-multicast-peers\n"
        "log-file=stdout\n",
    )


def main() -> None:
    from huggingface_hub import snapshot_download

    turn_config()
    kokoro = MODELS / "kokoro"
    download(f"{KOKORO_BASE}/kokoro-v1.0.onnx", kokoro / "kokoro-v1.0.onnx", 200_000_000)
    download(f"{KOKORO_BASE}/voices-v1.0.bin", kokoro / "voices-v1.0.bin", 20_000_000)
    print("Downloading Qwen weights if needed (~30 GB on first run)...", flush=True)
    snapshot_download(
        repo_id=f"Qwen/{QWEN}",
        local_dir=MODELS / QWEN,
    )
    print("Models and TURN credentials ready.", flush=True)


if __name__ == "__main__":
    main()
