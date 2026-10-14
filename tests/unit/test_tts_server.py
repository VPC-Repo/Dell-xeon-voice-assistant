"""Contract tests for the TTS server against the mock backend."""

import numpy as np
from fastapi.testclient import TestClient
from tts_server.app import build_app
from tts_server.backends import MockBackend
from tts_server.resample import StreamResampler


def make_client() -> TestClient:
    return TestClient(build_app(MockBackend()))


class TwoVoiceBackend(MockBackend):
    """Mock with two named voices at different native rates."""

    name = "twovoice"
    default_voice = "alpha"

    def voices(self):
        return ["alpha", "beta"]

    def sample_rate_for(self, voice):
        return 16000 if voice == "beta" else 24000

    async def stream(self, text, voice):
        self.last_voice = voice  # noqa: B010 - test instrumentation
        rate = self.sample_rate_for(voice)
        yield b"\x01\x00" * (rate // 10)  # 100 ms


def test_health_reports_backend():
    with make_client() as client:
        body = client.get("/health").json()
        assert body["backend"] == "mock"
        assert body["native_sample_rate"] == 24000


def test_voices_endpoint_and_selection():
    backend = TwoVoiceBackend()
    with TestClient(build_app(backend)) as client:
        listing = client.get("/voices").json()
        assert listing == {"voices": ["alpha", "beta"], "default": "alpha"}

        resp = client.post("/synthesize", json={"text": "hi", "voice": "beta", "sample_rate": 16000})
        assert resp.status_code == 200
        assert backend.last_voice == "beta"
        # Unknown voice falls back to the default rather than erroring.
        resp = client.post("/synthesize", json={"text": "hi", "voice": "nope", "sample_rate": 24000})
        assert resp.status_code == 200
        assert backend.last_voice == "alpha"


def test_synthesize_streams_pcm_at_requested_rate():
    with make_client() as client:
        native = client.post(
            "/synthesize",
            json={"text": "hello world test", "sample_rate": 24000},
        )
        assert native.status_code == 200
        resampled = client.post(
            "/synthesize",
            json={"text": "hello world test", "sample_rate": 48000},
        )
        # Same audio at double the rate: roughly double the bytes.
        ratio = len(resampled.content) / len(native.content)
        assert 1.9 < ratio < 2.1
        assert len(native.content) % 2 == 0


def test_rejects_bad_format_and_empty_text():
    with make_client() as client:
        assert (
            client.post("/synthesize", json={"text": "hi", "format": "mp3"}).status_code == 400
        )
        assert client.post("/synthesize", json={"text": ""}).status_code == 422


def test_resampler_preserves_signal_shape():
    src_rate, dst_rate = 22050, 24000
    t = np.arange(src_rate) / src_rate  # 1 s of 440 Hz sine
    signal = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    rs = StreamResampler(src_rate, dst_rate)
    out = b""
    for i in range(0, len(signal), 1000):  # stream in uneven chunks
        out += rs.push(signal[i : i + 1000].tobytes())
    out += rs.flush()
    result = np.frombuffer(out, dtype=np.int16)
    # Duration preserved within one chunk of slack.
    assert abs(len(result) - dst_rate) < 100
    # Amplitude preserved (it's still a sine of the same magnitude).
    assert 9000 < result.max() <= 10001
