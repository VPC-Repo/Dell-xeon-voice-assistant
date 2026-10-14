"""Start the packaged services with mock adapters and no external network."""

import json
import os
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def wait_ready(url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                return json.load(response)
        except URLError:
            time.sleep(0.1)
    raise RuntimeError(f"Service did not start: {url}")


def main():
    # Import the packaged native speech libraries without loading models.
    import kokoro_onnx  # noqa: F401
    import moonshine_voice  # noqa: F401

    env = os.environ | {
        "ADAPTER_MODE": "mock",
        "STUN_URLS": "",
        "AUTH_TOKEN": "",
        "BIND_HOST": "127.0.0.1",
        "BIND_PORT": "8080",
        "TTS_BACKEND": "mock",
        "TTS_BIND_HOST": "127.0.0.1",
        "TTS_BIND_PORT": "8082",
    }
    processes = []
    try:
        for command in ("voice-gateway", "tts-server"):
            processes.append(subprocess.Popen([command], env=env))
        assert wait_ready("http://127.0.0.1:8080/api/ready")["ready"]
        assert wait_ready("http://127.0.0.1:8082/health")["backend"] == "mock"
        with urlopen("http://127.0.0.1:8080/", timeout=2) as response:
            assert b"app.js?v=" in response.read()
        with urlopen(
            Request("http://127.0.0.1:8080/api/sessions", method="POST"), timeout=2
        ) as response:
            sid = json.load(response)["session_id"]
        with urlopen(
            Request(f"http://127.0.0.1:8080/api/sessions/{sid}", method="DELETE"), timeout=2
        ) as response:
            assert response.status == 204
        request = Request(
            "http://127.0.0.1:8082/synthesize",
            data=json.dumps({"text": "Hello there."}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            assert len(response.read()) > 0
        print("Container startup, UI, signaling and mock synthesis passed.")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
