from gateway.orchestration.echo import is_probable_echo

RECENT = [
    "The forecast for tomorrow looks mostly sunny with a light breeze.",
    "Highs will be near seventy five degrees in the afternoon.",
]


def test_exact_echo_is_detected():
    assert is_probable_echo(
        "The forecast for tomorrow looks mostly sunny with a light breeze.", RECENT
    )


def test_lossy_stt_echo_is_detected():
    # STT of leaked playback: dropped words, no punctuation, case changes.
    assert is_probable_echo("forecast tomorrow looks mostly sunny light breeze", RECENT)
    assert is_probable_echo("highs will be near seventy five degrees", RECENT)


def test_real_user_speech_passes():
    assert not is_probable_echo("What about the weekend, will it rain?", RECENT)
    assert not is_probable_echo("Tell me a joke about penguins.", RECENT)


def test_short_answers_pass_unless_verbatim():
    # "yes" / "okay" must never be eaten just because the assistant used
    # those words somewhere.
    assert not is_probable_echo("yes please", RECENT)
    # But a verbatim short phrase from the playback is an echo.
    assert is_probable_echo("light breeze", RECENT)


def test_disabled_and_empty_cases():
    assert not is_probable_echo("anything", RECENT, threshold=0)
    assert not is_probable_echo("anything", [])
    assert not is_probable_echo("", RECENT)
