"""Credential creation and resumable model downloads without external services."""

import importlib.util
import stat
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "compose_models.py"
_spec = importlib.util.spec_from_file_location("compose_models", _SRC)
compose_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compose_models)


def test_turn_credentials_are_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(compose_models, "RUNTIME", tmp_path)
    compose_models.turn_config()
    password = (tmp_path / "turn-password").read_text().strip()
    assert len(password) == 64
    assert f"TURN_PASSWORD={password}" in (tmp_path / "turn.env").read_text()
    assert f"user=voice:{password}" in (tmp_path / "turnserver.conf").read_text()
    for path in tmp_path.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    compose_models.turn_config()
    assert (tmp_path / "turn-password").read_text().strip() == password


def test_invalid_persisted_password_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(compose_models, "RUNTIME", tmp_path)
    (tmp_path / "turn-password").write_text("invalid\nINJECTED=value")
    with pytest.raises(RuntimeError, match="Invalid stored TURN password"):
        compose_models.turn_config()
    assert not (tmp_path / "turn.env").exists()


def test_failed_download_keeps_existing_file_and_removes_partial(tmp_path, monkeypatch):
    target = tmp_path / "model.bin"
    target.write_bytes(b"old")

    def truncated(url, temporary):
        Path(temporary).write_bytes(b"short")

    monkeypatch.setattr(compose_models, "urlretrieve", truncated)
    with pytest.raises(RuntimeError, match="download too small"):
        compose_models.download("https://models.example/model.bin", target, 10)
    assert target.read_bytes() == b"old"
    assert not target.with_suffix(".bin.partial").exists()
